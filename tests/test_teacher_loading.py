import unittest
import torch
from generate_labels import materialize_state_dict
from pretrain.model.TemporalTransformer import TemporalTransformer

class TeacherLoadingTests(unittest.TestCase):
    def model(self, device=None):
        return TemporalTransformer(768,1024,512,8,2,16,1,0.,False,self.a,self.v,device=device)
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        self.a=torch.randn(25,512);self.v=torch.randn(25,768)
    def test_meta_loading_preserves_every_tensor_and_predictions(self):
        reference=self.model().eval();state=reference.state_dict()
        restored=materialize_state_dict(self.model('meta'),state).eval()
        for k,v in state.items():
            self.assertTrue(torch.equal(v,restored.state_dict()[k]))
            self.assertEqual(v.data_ptr(),restored.state_dict()[k].data_ptr())
        a=torch.randn(2,10,512);v=torch.randn(2,10,768)
        with torch.no_grad():
            expected=reference(a,v);actual=restored(a,v)
        for x,y in zip(expected,actual):self.assertTrue(torch.equal(x,y))
    def test_wrong_checkpoint_is_rejected(self):
        state=self.model().state_dict();state.pop(next(iter(state)))
        with self.assertRaisesRegex(ValueError,'keys'):materialize_state_dict(self.model('meta'),state)
        state=self.model().state_dict();state['a_logit_scale']=torch.ones(3)
        with self.assertRaisesRegex(ValueError,'shape'):materialize_state_dict(self.model('meta'),state)

if __name__=='__main__':unittest.main()
