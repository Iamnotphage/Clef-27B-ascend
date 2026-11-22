#!/usr/bin/env python3
"""Correct private generated-API tiling capacity; never changes kernel ELF.

The public composite L2 executor accepts Phase6's appended tiling trailer.
NNOPBASE's generated L0 API instead initializes RawTilingData from opParaSize,
which the compiler derived from the first GET_TILING_DATA struct (584/592 B).
CANN 9.2.1 OpImplRegisterV2::Tiling's default maximum is 2048 B. Retain that
registered capacity for these two private diagnostics, including the trailer.
The actual serialized length and fields remain chosen by the original host.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--record', type=Path)
    a=p.parse_args(); v=a.vendor.resolve()
    if v.name != 'clef40_kkt_transformer':
        p.error('only separately named private vendor is accepted')
    if a.record and a.record.exists():
        p.error('record must be new')
    records=[]
    for path in sorted(v.rglob('ClefGdnKkt*.json')):
        data=json.loads(path.read_text())
        if 'opParaSize' not in data:
            continue
        elf=path.with_suffix('.o')
        if not elf.is_file() or data['opParaSize'] not in (584,592,2048):
            p.error('unexpected private binary metadata '+str(path))
        before=hashlib.sha256(elf.read_bytes()).hexdigest()
        item=dict(path=str(path),original_capacity=data['opParaSize'],capacity=2048,
                  elf_sha256=before,metadata_before_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if a.apply:
            data['opParaSize']=2048
            path.write_text(json.dumps(data,indent=2)+'\n')
            assert hashlib.sha256(elf.read_bytes()).hexdigest()==before
        item['metadata_after_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(item)
    if len(records) != 8:
        p.error('expected exactly eight private signature metadata files')
    result=dict(status='applied' if a.apply else 'plan',registered_maximum=2048,
                kernel_changed=False,files=records)
    if a.record:
        a.record.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],count=len(records),kernel_changed=False)))


if __name__=='__main__':main()
