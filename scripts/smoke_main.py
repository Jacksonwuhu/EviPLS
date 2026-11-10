#!/usr/bin/env python3
"""Two-round CPU training check on two videos per split.

Uses the actual parser, both optimizers, validation selection, checkpoint I/O,
U-HNM and IPR. Only dataset length / epochs / batch size / workers are reduced.
"""
import argparse,json,os,sys,time
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import torch
import main_ipr
import reproduce
from dataloader import LLP_dataset
from nrep.data import build_llp_dataloaders
from teacher_train import bounded_runtime

class TwoVideos(LLP_dataset):
    def __len__(self):return min(2,super().__len__())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--label-root',type=Path);p.add_argument('--timeout-seconds',type=int,default=600);args=p.parse_args()
    torch.set_num_threads(1)
    flags=['train-llp','--run-id',args.run_id,'--device','cpu','--epochs','1','--batch-size','2','--workers','0','--threads','1']
    if args.label_root:flags+=['--label-root',str(args.label_root)]
    plan=reproduce.build_plan(reproduce.parser().parse_args(flags));out=Path(plan['output_directories'][0]);out.mkdir(parents=True,exist_ok=False)
    record={'scope':'2 real videos per split; 1 epoch per phase; two rounds','state':'running','plan':plan}
    state=out/'smoke.json';state.write_text(json.dumps(record,indent=2)+'\n');started=time.monotonic()
    previous=Path.cwd();os.chdir(ROOT)
    try:
        with bounded_runtime(args.timeout_seconds),mock.patch.object(sys,'argv',plan['commands'][-1][1:]),mock.patch.object(main_ipr,'LLP_dataset',TwoVideos),mock.patch.object(main_ipr,'get_LLP_dataloader',lambda a:build_llp_dataloaders(a,dataset_cls=TwoVideos)):
            main_ipr.main()
        checkpoint=ROOT/'models'/args.run_id/(args.run_id+'_r2.best')
        state_dict=torch.load(checkpoint,map_location='cpu')
        if not {'model_audio','model_video'}<=state_dict.keys():raise ValueError('Missing dual-model checkpoint')
        from generate_labels import select_records
        for row in select_records('train',2):
            for modality in ('audio','visual'):
                file=out/'pseudo_labels/round_2'/modality/(row['video']+'.npy')
                import numpy as np
                array=np.load(file,allow_pickle=False)
                if array.shape!=(10,25) or not np.isfinite(array).all() or (array<0).any() or (array>1).any():raise ValueError(str(file))
        record.update(state='complete',checkpoint=str(checkpoint))
    except BaseException as exc:record.update(state='failed',error=repr(exc));raise
    finally:
        os.chdir(previous);record['elapsed_seconds']=time.monotonic()-started;state.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
