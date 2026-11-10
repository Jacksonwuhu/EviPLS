import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from reproduce import run_bounded

class BudgetTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','POSIX process-group contract')
    def test_timeout_stops_parent_and_worker_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pidfile=root/'worker.pid'
            code="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path('worker.pid').write_text(str(p.pid)); time.sleep(60)"
            with open(os.devnull,'w') as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_bounded([sys.executable,'-c',code],cwd=tmp,env=os.environ.copy(),stdout=log,timeout=2)
            self.assertTrue(pidfile.exists())
            status=Path('/proc')/pidfile.read_text()/'stat'
            if status.exists():self.assertIn(status.read_text().split()[2],['Z','X'])

if __name__=='__main__':unittest.main()
