#!/usr/bin/env python3
"""Install checksum-pinned upstream inputs and/or the Git LFS LLP asset bundle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REVISION = '71dcb419fb8ff88ce289b396470abdf8b597f27e'
UPSTREAM = {
 'UnAV_CLAP.zip': ('09c162c9efb0944ef25d2c84151533dd60bff3314eb4067b90f6da6a36d06484', 875118163, 'data/UnAV/feats_CLAP'),
 'UnAV_CLIP.zip': ('3987fa662687776c8cd68af78023038397b5cdfd1461972c72d800f553c1fd19', 1304627715, 'data/UnAV/feats_CLIP'),
}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2), b''): h.update(b)
 return h.hexdigest()

def safe_relative(name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or '\\' in name:
  raise ValueError(f'Unsafe archive path: {name}')
 return p

def download(url, destination, digest, size, deadline, ignore_proxy=False):
 if destination.exists():
  if destination.stat().st_size==size and sha256(destination)==digest: return
  raise ValueError(f'Existing download failed checksum: {destination}')
 partial=destination.with_suffix(destination.suffix+'.part')
 if partial.exists(): raise FileExistsError(f'Previous incomplete download: {partial}; inspect/remove before retrying')
 destination.parent.mkdir(parents=True,exist_ok=True)
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if ignore_proxy else urllib.request.build_opener()
 h=hashlib.sha256(); total=0
 with opener.open(url,timeout=60) as response, partial.open('xb') as f:
  while True:
   if time.monotonic()>deadline: raise TimeoutError('Download budget exhausted')
   block=response.read(4*1024**2)
   if not block: break
   total+=len(block)
   if total>size: raise ValueError('Download exceeds pinned byte size')
   h.update(block); f.write(block)
 if total!=size or h.hexdigest()!=digest: raise ValueError(f'Download checksum/size mismatch: {destination.name}')
 partial.rename(destination)

def extract_zip(archive,destination):
 if destination.exists(): raise FileExistsError(destination)
 destination.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=destination.parent,prefix='.extract-') as tmp:
  with zipfile.ZipFile(archive) as z:
   for info in z.infolist():
    relative=safe_relative(info.filename)
    if stat.S_ISLNK(info.external_attr>>16): raise ValueError('Archive symlink rejected')
    target=Path(tmp).joinpath(*relative.parts)
    if info.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(info) as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
  # Publish only after every file passed ZIP CRC verification.
  os.rename(tmp,destination)

def install_assets(root=ROOT):
 manifest=json.loads((root/'assets/manifest.json').read_text())
 for item in manifest['archives']:
  archive=root/'assets'/item['file']
  if not archive.is_file() or archive.stat().st_size!=item['bytes'] or sha256(archive)!=item['sha256']:
   raise ValueError(f'{archive}: missing/corrupt asset or Git LFS pointer; run git lfs pull')
  with tarfile.open(archive,'r:gz') as tar:
   members=tar.getmembers()
   for member in members:
    relative=safe_relative(member.name)
    if not member.isfile(): raise ValueError('Asset archives must contain only regular files')
    target=root.joinpath(*relative.parts)
    if any(p.is_symlink() for p in [target,*target.parents] if p!=root.parent): raise ValueError('Refusing symlink destination')
    if target.exists():
     source=tar.extractfile(member)
     if hashlib.sha256(source.read()).hexdigest()!=sha256(target): raise ValueError(f'Refusing to overwrite different input: {target}')
     continue
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
  print('Installed',item['file'],flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--unav',action='store_true');p.add_argument('--assets',action='store_true')
 p.add_argument('--base-url',default='https://huggingface.co',help='Optional compatible mirror; pinned SHA256 is always enforced')
 p.add_argument('--ignore-proxy',action='store_true')
 p.add_argument('--timeout-hours',type=float,default=2)
 p.add_argument('--cache',type=Path,default=ROOT/'.cache/downloads')
 args=p.parse_args()
 if not (args.unav or args.assets): p.error('select --unav and/or --assets')
 if not 0<args.timeout_hours<=24:p.error('timeout-hours must be in (0,24]')
 if args.assets: install_assets()
 if args.unav:
  deadline=time.monotonic()+args.timeout_hours*3600
  report={'revision':REVISION,'state':'running','files':[],'max_download_bytes':sum(v[1] for v in UPSTREAM.values()),'timeout_hours':args.timeout_hours}
  args.cache.mkdir(parents=True,exist_ok=True)
  record=args.cache/('download-'+str(time.time_ns())+'.json')
  try:
   for name,(digest,size,relative) in UPSTREAM.items():
    dest=ROOT/relative
    if dest.exists(): raise FileExistsError(f'Refusing existing UnAV feature directory: {dest}')
    url=args.base_url.rstrip('/')+'/datasets/NTUBarista/uwav_features/resolve/'+REVISION+'/'+name
    bundled=ROOT/'assets'/name
    archive=bundled if bundled.is_file() and bundled.stat().st_size==size else args.cache/name
    print('Preparing',name,flush=True)
    download(url,archive,digest,size,deadline,args.ignore_proxy)
    extract_zip(archive,dest)
    report['files'].append({'name':name,'sha256':digest,'bytes':size,'url':url})
    record.write_text(json.dumps(report,indent=2)+'\n')
    print('Verified and extracted',name,flush=True)
   report['state']='complete'
  except BaseException as exc:
   report.update(state='failed',error=repr(exc));raise
  finally:record.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
