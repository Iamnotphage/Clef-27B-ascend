#!/usr/bin/env python3
"""Check the fixed Fusion tile ownership, ring lifetime and byte budget on CPU."""
import json
from math import ceil


def tiles(tokens, width):
    # CATLASS GemmIdentityBlockSwizzle<3,1>, fixed M128/N256.
    mr, nr = ceil(tokens / 128), width // 256
    coords = []
    for task in range(mr * nr):
        panel, within = divmod(task, 3 * mr)
        columns = min(3, nr - 3 * panel)
        m, n = divmod(within, columns)
        coords.append((mr - 1 - m if panel % 2 else m, 3 * panel + n))
    assert set(coords) == {(m, n) for m in range(mr) for n in range(nr)}
    return coords


def check(tokens):
    groups, ring_size = 24, 2
    tile_rows, tile_cols, gate_width, hidden = 128, 256, 8704, 5120
    ring_elements = groups * ring_size * tile_rows * tile_cols * 2
    coords = tiles(tokens, gate_width)
    produced, consumed = set(), set()
    for group in range(groups):
        ordinal = 0
        group_tiles = coords[group::groups]
        for i in range(0, len(group_tiles), ring_size):
            # Worst-case lead: both pair slots are published before either read.
            pending = {}
            for m, n in group_tiles[i:i+ring_size]:
                base = (group * ring_size + (ordinal % ring_size)) * tile_rows * tile_cols * 2
                assert base not in pending
                pending[base] = (m, n)
                assert (m, n) not in produced
                produced.add((m, n))
                ordinal += 1
            for base, (m, n) in pending.items():
                rows = min(tile_rows, tokens - m * tile_rows)
                assert base + (rows - 1) * 512 + 512 <= ring_elements
                lane_tiles = []
                for lane in range(2):
                    start = lane * 64
                    end = min(rows, start + 64)
                    for row in range(start, end, 32):
                        for index in range(row, min(row + 32, end)):
                            lane_tiles.append(index)
                            key = (m * tile_rows + index, n)
                            assert key not in consumed
                            consumed.add(key)
                assert lane_tiles == list(range(rows))
            # AIC does not overwrite a previous batch until both lanes ack.
    assert len(consumed) == tokens * (gate_width // tile_cols)
    down = tiles(tokens, hidden)
    # Each down tile sees every inner-256 activation slice only after publication.
    for m, _ in down:
        for row in range(m * tile_rows, min((m + 1) * tile_rows, tokens)):
            assert all((row, n) in consumed for n in range(gate_width // tile_cols))
    return dict(tokens=tokens, gate_pairs=len(coords), down_tiles=len(down),
                ring_bytes=ring_elements * 2, activation_bytes=tokens * gate_width * 2,
                gate_up_logical_store_read_bytes=tokens * gate_width * 2 * 2 * 2,
                activation_logical_write_one_read_bytes=tokens * gate_width * 2 * 2,
                streaming_activations_per_down_row=hidden // tile_cols)


def main():
    kib = 1024
    l1 = 2 * (128 * 256 + 256 * 256) * 2
    l0a, l0b, l0c = 2 * 128 * 64 * 2, 2 * 64 * 256 * 2, 128 * 256 * 4
    ub = 32 * 256 * (2 * 2 + 2 + 3 * 4)
    assert (l1, l0a, l0b, l0c, ub) == (384*kib, 32*kib, 64*kib, 128*kib, 144*kib)
    rows = [check(m) for m in (1, 64, 65, 127, 128, 129, 307, 967, 1595)]
    print(json.dumps(dict(status="cpu_contract_passed",
                          local_bytes=dict(l1=l1, l0a=l0a, l0b=l0b, l0c=l0c, ub=ub),
                          points=rows), indent=2))


if __name__ == "__main__":
    main()
