import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from nrep.validation import _check_artifact, validate_pipeline
from prepare_data import extract_zip, safe_relative

class DataIntegrityTests(unittest.TestCase):
    def test_checks_every_array_not_just_one_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.save(root/'a.npy', np.zeros((10,25),np.float32))
            np.save(root/'b.npy', np.full((10,25),np.nan,np.float32))
            report=_check_artifact('audio pseudo-label',root,['a','b'],(10,25))
            self.assertFalse(report.ok)
            self.assertEqual(report.invalid_files,1)
            np.save(root/'b.npy',np.full((10,25),2.,np.float32))
            self.assertFalse(_check_artifact('audio pseudo-label',root,['a','b'],(10,25)).ok)
    def test_missing_file_and_wrong_modality_dimensions_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.save(root/'a.npy',np.zeros((10,512),np.float32))
            report=_check_artifact('student audio',root,['a','b'],(64,768))
            self.assertEqual((report.invalid_files,report.missing_files),(1,1))
    def test_feature_key_collision_across_splits_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir()
            for name in ['AVVP_train.csv','AVVP_val_pd.csv','AVVP_test_pd.csv']:
                (root/'data'/name).write_text('filename\tevent_labels\n12345678901_1_2\tSpeech\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'): validate_pipeline(root)
    def test_zip_traversal_does_not_publish_partial_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);z=root/'evil.zip';out=root/'features'
            with zipfile.ZipFile(z,'w') as archive:archive.writestr('../escaped.txt','bad')
            with self.assertRaises(ValueError):extract_zip(z,out)
            self.assertFalse(out.exists());self.assertFalse((root/'escaped.txt').exists())
    def test_valid_zip_install_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);z=root/'good.zip';out=root/'features'
            with zipfile.ZipFile(z,'w') as archive:archive.writestr('segment_feats/example','valid')
            extract_zip(z,out)
            self.assertEqual((out/'segment_feats/example').read_text(),'valid')
            with self.assertRaises(FileExistsError):extract_zip(z,out)

if __name__=='__main__':unittest.main()
