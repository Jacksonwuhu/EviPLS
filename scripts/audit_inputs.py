#!/usr/bin/env python3
"""Validate LLP and UnAV features, labels, category mappings and file hashes."""
import argparse, csv, gzip, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from prepare_data import sha256
from generate_labels import select_records,teacher_paths
from nrep.validation import validate_pipeline
from teacher_train import preflight

def audit(root=ROOT):
    report={'scope':'all LLP input values; all UnAV train/validation values', 'llp':{},'teacher_llp':{},'unav':{}}
    llp=validate_pipeline(root)
    if not llp.ok:raise ValueError('LLP student inputs failed validation')
    report['llp']={'videos':llp.videos,'split_sizes':llp.split_sizes,'checked_arrays':sum(x.expected_files for x in llp.artifacts)}
    records=select_records('all',None,root)
    for folder,dim in [('feats_CLAP',512),('feats_CLIP',768)]:
        for row in records:
            path=root/'pretrain/data/LLP'/folder/'segment_feats'/(row['video']+'.npy')
            value=np.load(path,allow_pickle=False)
            if value.shape!=(10,dim) or not np.isfinite(value).all():raise ValueError(f'Invalid teacher input: {path}')
        report['teacher_llp'][folder]={'files':len(records),'shape':[10,dim]}
    for name,path in teacher_paths(root).items():
        if not path.is_file():raise FileNotFoundError(path)
        report['teacher_llp'][name]={'sha256':sha256(path)}
    report['unav']=preflight(root/'data/UnAV')
    report['status']='passed'
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.output and a.output.exists():raise FileExistsError(a.output)
    report=audit()
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'status':report['status'],'LLP':report['llp'],'UnAV':report['unav']['split_counts']},indent=2))
if __name__=='__main__':main()
