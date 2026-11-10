#!/usr/bin/env python3
"""One real UnAV optimizer step with a reduced backbone; not a formal teacher run."""
import argparse,copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import numpy as np
import torch
import teacher_train as teacher
from pretrain.dataloader import build_dataloader
from pretrain.model import build_model

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/UnAV';document=json.loads((source/teacher.INPUT_LAYOUT['annotations']).read_text())
    selected={}
    for split,count in [('train',2),('validation',1)]:
        candidates=sorted([(item['duration'],key) for key,item in document['database'].items() if item['subset'].lower()==split])
        selected.update({key:document['database'][key] for _,key in candidates[:count]})
    subset=output/'diagnostic_subset';subset.mkdir()
    (subset/teacher.INPUT_LAYOUT['annotations']).write_text(json.dumps({'verification_only_subset':True,'database':selected}))
    for folder in ('feats_CLAP','feats_CLIP'):(subset/folder).symlink_to(source/folder,target_is_directory=True)
    config=teacher.prepare_args(teacher.parser().parse_args(['--data-root',str(subset),'--output',str(output/'unused'),'--device','cpu']))
    config.hidden_dim=8;config.nhead=2;config.ff_dim=16;config.num_layers=1;config.dropout=0;config.batch_size=2
    torch.set_num_threads(1);torch.manual_seed(1000)
    audio=torch.from_numpy(np.load(source/teacher.INPUT_LAYOUT['clap_events'],allow_pickle=False))
    visual=torch.from_numpy(np.load(source/teacher.INPUT_LAYOUT['clip_events'],allow_pickle=False))
    model=build_model(config,config.model,visual,audio)
    optimizer,scheduler=teacher.make_optimizer_scheduler(config,model)
    original=model.proj_a.layers[0].weight.detach().clone()
    with teacher.bounded_runtime(300):
        train=build_dataloader(config,teacher.FeatureOnlyUnAVDataset('train',subset),'train')
        validation=build_dataloader(config,teacher.FeatureOnlyUnAVDataset('val',subset),'val')
        loss=teacher.train_epoch(config,model,train,optimizer,torch.device('cpu'))
        scheduler.step()
        scores,val_loss=teacher.evaluate_validation(config,model,validation,torch.device('cpu'))
    if not np.isfinite(loss['loss_all']) or torch.equal(original,model.proj_a.layers[0].weight):raise ValueError('No finite optimizer update')
    report={'state':'complete','scope':'2 real UnAV train videos + 1 validation video; reduced hidden_dim=8, one layer; not the full teacher', 'train_loss':loss,'validation_loss':val_loss,'validation_metrics':scores,'video_ids':list(selected),'parameter_update_verified':True}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
