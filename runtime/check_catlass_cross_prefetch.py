#!/usr/bin/env python3
"""CPU contract checks for fixed eec373fb, fixed K5120 and two L1 slots.

This finite-state model checks the selected driver schedule, not hardware
ordering or template expansion. Official sync-analyzer candidates stay in raw.
"""
import argparse
import hashlib
import json
from pathlib import Path


def verify_source(root):
    pin = json.loads((Path(__file__).parent/'ascendc/clef_mlp_cross_prefetch/catlass-source.json').read_text())
    for name, digest in pin['contract_headers_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    return pin['commit']


def coordinates(tokens):
    # Official GemmIdentityBlockSwizzle<3,1>, M128/N256.
    rows, cols = (tokens+127)//128, 68
    result = []
    for task in range(rows*cols):
        panel, within = divmod(task, 3*rows)
        width = min(3, cols-panel*3)
        m, n = within//width, panel*3+within % width
        if panel % 2: m = rows-m-1
        result.append((m, n))
    assert len(set(result)) == rows*cols
    assert set(result) == {(m,n) for m in range(rows) for n in range(cols)}
    return result


def simulate(tokens, preload):
    coords = coordinates(tokens)
    reads = first_at_entry = cross = 0
    for core in range(24):
        tasks = list(range(core, len(coords), 24))
        free, ready, slot = [True,True], [None,None], 0
        def load(which, task, k):
            nonlocal reads
            assert free[which] and ready[which] is None, (core,task,k,which)
            free[which] = False; ready[which] = (task,k); reads += 1
        for ordinal, task in enumerate(tasks):
            for k in range(20):
                if k == 0 and (ordinal == 0 or not preload):
                    load(slot,task,0); first_at_entry += 1
                nxt = (slot+1)%2
                if k != 19:
                    load(nxt,task,k+1)
                elif preload and ordinal+1 < len(tasks):
                    load(nxt,tasks[ordinal+1],0); cross += 1
                assert ready[slot] == (task,k), (core,task,k,slot,ready)
                ready[slot] = None; free[slot] = True
                slot = nxt
        assert all(free) and ready == [None,None]
    assert reads == len(coords)*20
    return dict(m=tokens, tiles=len(coords), k_slices=20, gm_ab_tile_loads=reads,
                first_at_entry=first_at_entry, cross_first_k_preloads=cross,
                all_slots_released=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catlass-root', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    commit = verify_source(a.catlass_root)
    npoints = 0
    for n in range(17408):
        for k in (0,1,15,16,31,63,64,255,256,5119):
            nz_weight = ((k//16)*(17408//16)+n//16)*256+(n%16)*16+k%16
            nz_transposed = (k//16)*(17408*16)+(n//16)*256+(k%16)+(n%16)*16
            assert nz_weight == nz_transposed
            npoints += 1
    rows = []
    for m in (967,307,1595):
        off, on = simulate(m,False), simulate(m,True)
        assert off['gm_ab_tile_loads'] == on['gm_ab_tile_loads']
        rows.append(dict(off=off,on=on))
    out = dict(status='CPU_CONTRACT_PASS', catlass_commit=commit,
               nz_transposed_offset_points=npoints,l1_bytes=393216,
               l0a_double_bytes=32768,l0b_double_bytes=65536,l0c_bytes=131072,
               schedule=rows,device_numeric='NOT_RUN',
               boundary='finite-state contract only; regex sync candidates not resolved by this model')
    if a.output: a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__': main()
