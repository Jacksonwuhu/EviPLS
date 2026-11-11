import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile
import prepare_data


class InstallationResumeTests(unittest.TestCase):
    def test_zip_retry_checks_existing_files_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'features.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('segment_feats/a.npy', b'feature')
            dest = root / 'installed'
            prepare_data.extract_zip(archive, dest)
            prepare_data.verify_zip_install(archive, dest)
            (dest / 'segment_feats/a.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed installed'):
                prepare_data.verify_zip_install(archive, dest)

    def test_interrupted_tar_install_can_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'assets').mkdir()
            archive = root / 'assets/data.tar.gz'
            payload = b'complete feature'
            with tarfile.open(archive, 'w:gz') as tar:
                info = tarfile.TarInfo('data/features/a.npy')
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            manifest = {'archives': [{'file': archive.name, 'bytes': archive.stat().st_size,
                                     'sha256': prepare_data.sha256(archive)}]}
            (root / 'assets/manifest.json').write_text(json.dumps(manifest))
            def interrupted(src, dst):
                dst.write(src.read(3))
                raise OSError('interrupted write')
            with mock.patch.object(prepare_data.shutil, 'copyfileobj', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    prepare_data.install_assets(root)
            target = root / 'data/features/a.npy'
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])
            prepare_data.install_assets(root)
            prepare_data.install_assets(root)
            self.assertEqual(target.read_bytes(), payload)
            target.write_bytes(b'different input')
            with self.assertRaisesRegex(ValueError, 'different input'):
                prepare_data.install_assets(root)


if __name__ == '__main__':
    unittest.main()
