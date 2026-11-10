#!/usr/bin/env python3
"""Verify delivered archives or every installed LLP input against SHA256."""
import argparse,gzip,json
from pathlib import Path
from prepare_data import ROOT,sha256,safe_relative

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed',action='store_true',help='also verify all extracted LLP data/weight files')
    parser.add_argument('--unav-cache',type=Path,help='Explicit local location of the two original UnAV ZIPs (default: assets/)')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    checked=0
    for item in manifest['archives']:
        path=ROOT/'assets'/item['file']
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
            raise ValueError(f'Asset missing or changed: {path}; fetch Git LFS payloads first')
        checked+=1
    unav=json.loads((ROOT/'assets/unav_manifest.json').read_text())
    for item in unav['archives']:
        path=(args.unav_cache or ROOT/'assets')/item['file']
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
            raise ValueError(f'UnAV archive missing or changed: {path}')
        checked+=1
    if args.installed:
        with gzip.open(ROOT/'assets/file_manifest.json.gz','rt') as handle:files=json.load(handle)
        for item in files:
            path=ROOT.joinpath(*safe_relative(item['path']).parts)
            if not path.is_file() or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
                raise ValueError(f'Installed input missing or changed: {path}')
            checked+=1
    print(json.dumps({'status':'passed','checked_files':checked,'installed_checked':args.installed}))
if __name__=='__main__':main()
