"""CPU contract checks for the connected teacher -> labels -> parser workflow."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import reproduce
import generate_labels as labels
from calibrate_teacher import dense_labels


class TeacherPipelineTests(unittest.TestCase):
    def plan(self, action, *flags):
        args = reproduce.parser().parse_args([action, '--run-id', 'new_run', *flags])
        with tempfile.TemporaryDirectory() as folder:
            return reproduce.build_plan(args, Path(folder))

    def test_teacher_is_part_of_launcher(self):
        plan = self.plan('train-teacher', '--data-root', 'data/UnAV', '--check-only')
        command = plan['commands'][0]
        self.assertEqual(command[1], 'teacher_train.py')
        self.assertIn('--check-only', command)
        self.assertEqual(command[command.index('--seed') + 1], '1000')
        self.assertTrue(command[command.index('--output') + 1].endswith('/artifacts/new_run/teacher'))

    def test_generation_uses_new_teacher_and_thresholds(self):
        plan = self.plan('generate-labels', '--checkpoint', 'artifacts/new_teacher/teacher/best.pt',
                         '--threshold-dir', 'artifacts/calibration/thresholds', '--device', 'cpu')
        command = plan['commands'][0]
        self.assertEqual(command[1], 'generate_labels.py')
        self.assertEqual(command[command.index('--split') + 1], 'all')
        self.assertIn('--threshold-dir', command)
        self.assertEqual(plan['device'], 'cpu')

    def test_generated_labels_reach_training_and_evaluation(self):
        for action in ('train-llp', 'evaluate-llp'):
            flags = ['--checkpoint', 'model.best'] if action.startswith('evaluate') else []
            plan = self.plan(action, '--label-root', 'artifacts/new_labels/labels', *flags)
            command = plan['commands'][-1]
            for arg, suffix in (('--a_pseudo_data_dir', '/audio'), ('--v_pseudo_data_dir', '/visual')):
                self.assertTrue(command[command.index(arg) + 1].endswith('/artifacts/new_labels/labels' + suffix))

    def test_new_teacher_cannot_silently_use_old_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = labels.teacher_paths(root, 'new.pt')
            with self.assertRaisesRegex(ValueError, 'new teacher requires'):
                labels.validate_threshold_source(paths, root, None)

    def test_frozen_teacher_with_explicit_thresholds_is_not_exempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = labels.teacher_paths(root, thresholds='new_thresholds')
            with self.assertRaisesRegex(ValueError, 'calibration manifest'):
                labels.validate_threshold_source(paths, root, 'new_thresholds')

    def test_threshold_lineage_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'new.pt').write_bytes(b'fixture checkpoint')
            (root / 'thresholds').mkdir()
            paths = labels.teacher_paths(root, 'new.pt', 'thresholds')
            for key in ('audio_thresholds', 'visual_thresholds'):
                np.save(paths[key], np.zeros(25, dtype=np.float32))
            record = dict(state='complete', split='val', checkpoint_sha256=labels.sha256(paths['checkpoint']),
                          threshold_sha256={k: labels.sha256(paths[k]) for k in ('audio_thresholds', 'visual_thresholds')})
            manifest = root / 'thresholds/manifest.json'
            manifest.write_text(json.dumps(record))
            labels.validate_threshold_source(paths, root, 'thresholds')
            np.save(paths['audio_thresholds'], np.ones(25, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'changed since calibration'):
                labels.validate_threshold_source(paths, root, 'thresholds')

    def test_calibration_only_consumes_validation_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gt.tsv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['filename', 'event_labels', 'onset', 'offset'])
                writer.writerow(['val_id', 'Speech', 1, 3])
                writer.writerow(['test_id', 'Car', 0, 10])
            arrays = dense_labels(path, [{'filename': 'val_id'}])
            self.assertEqual(set(arrays), {'val_id'})
            self.assertEqual(arrays['val_id'].sum(), 2)
            np.testing.assert_array_equal(arrays['val_id'][1:3, 0], [1, 1])


if __name__ == '__main__':
    unittest.main()

class FrozenTeacherLineageTests(unittest.TestCase):
    def test_replacing_checkpoint_under_same_filename_cannot_reuse_frozen_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);paths=labels.teacher_paths(root)
            for path in paths.values():
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'known frozen bytes')
            (root/'assets').mkdir()
            (root/'assets/frozen_teacher.json').write_text(json.dumps({'sha256':{k:labels.sha256(p) for k,p in paths.items()}}))
            labels.validate_threshold_source(paths,root,None)
            paths['checkpoint'].write_bytes(b'a different model')
            with self.assertRaisesRegex(ValueError,'frozen teacher input changed'):
                labels.validate_threshold_source(paths,root,None)

class AnnotationCompatibilityTests(unittest.TestCase):
    def test_known_canonical_reversed_interval_matches_original_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'audio.tsv'
            path.write_text('filename\tevent_labels\tonset\toffset\n8DF2PxdT5tE_390_400\tCheering\t9\t0\n')
            anomalies=[]
            with self.assertWarns(UserWarning):
                result=dense_labels(path,[{'filename':'8DF2PxdT5tE_390_400'}],anomalies)
            self.assertEqual(result['8DF2PxdT5tE_390_400'].sum(),0)
            self.assertEqual(len(anomalies),1)
            path.write_text('filename\tevent_labels\tonset\toffset\nnew_video\tCheering\t9\t0\n')
            with self.assertRaises(ValueError):dense_labels(path,[{'filename':'new_video'}])
