#!/usr/bin/env python3
"""CPU-only integer address audit for the fixed TP2 paired BF16 NZ probe."""
import json

N,K,TILE_M,TILE_N,I=17408,5120,64,256,8704

def nz_offset(row,col):
    return ((col//16)*(N//16)+row//16)*256+(row%16)*16+col%16

def audit():
    # Every GEMM output column maps to the original gate-first packed weight.
    reorder=[(n//256)*128+n%128+(I if n%256>=128 else 0) for n in range(N)]
    assert sorted(reorder)==list(range(N))
    checked=0
    for start in range(0,N,TILE_N):
        for row in (0,15,16,127,128,239,255):
            for col in (0,15,16,511,512,5103,5119):
                assert nz_offset(start+row,col)==start*16+nz_offset(row,col)
                checked+=1
        assert reorder[start:start+128]==list(range(start//2,start//2+128))
        assert reorder[start+128:start+256]==list(range(I+start//2,I+start//2+128))
    # Local ND copy has 64 rows of 256 BF16 elements, then two halves of 128.
    for row in range(TILE_M):
        for col in range(128):
            gate=(row*16)*16+col # DataCopy blockLen=8, srcStride=8, 32B blocks.
            up=gate+128
            assert gate==row*TILE_N+col and up==row*TILE_N+128+col
    capacity=TILE_M*TILE_N
    bad_last=(TILE_M-1)*N+TILE_N-1
    good_last=(TILE_M-1)*TILE_N+TILE_N-1
    assert bad_last>=capacity and good_last==capacity-1
    return dict(nz_address_checks=checked,gate_order='SiLU(first_half)*second_half',
                paired_columns_bijective=True,local_c_capacity_elements=capacity,
                v1_default_orgN_last_element=bad_last,v2_dense_last_element=good_last,
                first_code_defect='GEMM GetTensorC ND destination stride before epilogue',
                limits='Integer address proof only; device stages/rounding/synchronization still require numerical replay')

if __name__=='__main__':print(json.dumps(audit(),indent=2))
