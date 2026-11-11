import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PipelineScriptTests(unittest.TestCase):
    def test_input_failure_stops_before_teacher_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'python'
            log = root / 'calls.txt'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nexit 7\n')
            fake.chmod(0o755)
            result = subprocess.run(['bash', str(ROOT / 'scripts/run_full_pipeline.sh')],
                                    env=dict(os.environ, DRY_RUN='0', RUN_ID='test_failed_preflight',
                                             PYTHON_BIN=str(fake), CALL_LOG=str(log)),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(log.read_text().splitlines(), ['scripts/audit_inputs.py'])

    def test_preview_connects_all_five_stages_from_another_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(['bash', str(ROOT / 'scripts/run_full_pipeline.sh')],
                                    cwd=cwd, env=dict(os.environ, DRY_RUN='1', RUN_ID='test_preview_pipeline',
                                                     PYTHON_BIN=sys.executable),
                                    check=True, capture_output=True, text=True)
        decoder = json.JSONDecoder()
        plans = []
        remaining = result.stdout.strip()
        while remaining:
            value, end = decoder.raw_decode(remaining)
            plans.append(value)
            remaining = remaining[end:].strip()
        self.assertEqual([p['action'] for p in plans],
                         ['train-teacher', 'calibrate-teacher', 'generate-labels', 'train-llp', 'evaluate-llp'])
        self.assertEqual(plans[1]['checkpoint'], plans[2]['checkpoint'])
        self.assertEqual(plans[3]['label_root'], plans[4]['label_root'])
        self.assertTrue(plans[4]['checkpoint'].endswith('test_preview_pipeline_main_r2.best'))
        for plan in plans:
            self.assertFalse(Path(plan['output_directories'][0]).exists())


if __name__ == '__main__':
    unittest.main()
