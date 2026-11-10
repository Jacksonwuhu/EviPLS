#!/usr/bin/env python3
"""Build bounded Git LFS tarballs from the audited inputs; never modify inputs."""
from pathlib import Path
import gzip, hashlib, json, sys, tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_data import sha256
GROUPS={
 'student_audio':['data/CLAP/features'],
 'student_visual':['data/CLIP/features','pretrain/data/LLP/feats/r2plus1d_18'],
 'teacher_llp':['pretrain/data/LLP/feats_CLAP','pretrain/data/LLP/feats_CLIP'],
 'labels_weights':['data/CLAP/a_segmentwise_pseudo_labels','data/CLIP/v_segmentwise_pseudo_labels',
                  'pretrain/checkpoints','weights/llp','data/CLAP/a_classwise_threhsolds.npy','data/CLAP/v_classwise_threhsolds.npy'],
}
def main():
 output=ROOT/'assets';output.mkdir(exist_ok=False)
 report={'schema':1,'scope':'LLP teacher inputs, student inputs, frozen pseudo-labels and checkpoints; no raw videos', 'archives':[]}
 inventory=[]
 for group,folders in GROUPS.items():
  files=[]
  for folder in folders:
   p=ROOT/folder
   files.extend([p] if p.is_file() else sorted(p.rglob('*.npy' if group!='labels_weights' else '*')))
  files=[p for p in files if p.is_file() and p.suffix in ('.npy','.pt')]
  chunks=[[]]; size=0
  for p in files:
   if size+p.stat().st_size>900*1024**2 and chunks[-1]: chunks.append([]);size=0
   chunks[-1].append(p);size+=p.stat().st_size
  for i,chunk in enumerate(chunks):
   name=f'{group}_{i+1:02}.tar.gz';out=output/name
   with out.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
    for p in chunk:
     rel=p.relative_to(ROOT).as_posix()
     info=tar.gettarinfo(str(p),arcname=rel);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
     with p.open('rb') as src:tar.addfile(info,src)
     inventory.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha256(p),'archive':name})
   item={'file':name,'bytes':out.stat().st_size,'sha256':sha256(out),'files':len(chunk)}
   report['archives'].append(item);print(json.dumps(item),flush=True)
 (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 with gzip.open(output/'file_manifest.json.gz','wt') as f:json.dump(inventory,f)
 print('Total:',len(inventory),'files;',sum(x['bytes'] for x in report['archives']),'archive bytes',flush=True)
if __name__=='__main__':main()
