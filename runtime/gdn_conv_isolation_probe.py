#!/usr/bin/env python3
"""Probe request isolation of the AscendC causal-conv1d prefill path.

The model's GDN layers call ``causal_conv1d_fn`` once for the whole prefill
batch. This probe compares that batch call with one call per sequence under
zero and dirty convolution-cache slots, and checks the GDN state gather's
``has_initial_state=False`` contract.
"""

import json
import sys

import torch

sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
from fla_npu.ops.ascendc import causal_conv1d_fn  # noqa: E402
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states  # noqa: E402

DEV = "npu:0"
DIM = 5120
HEADS = 40
WIDTH = 4


def conv(x, weight, states, qsl, cache, has):
    return causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states=states,
        query_start_loc=qsl,
        cache_indices=cache,
        has_initial_state=has,
        activation="silu",
        pad_slot_id=-1,
        null_block_id=0,
        head_num=HEADS,
    )


def main():
    torch.npu.set_device(0)
    generator = torch.Generator(device="cpu").manual_seed(20261114)
    lengths = [606, 977, 430]
    total = sum(lengths)
    x = torch.randn(total, DIM, generator=generator).to(DEV, torch.bfloat16)
    weight = torch.randn(WIDTH, DIM, generator=generator).to(DEV, torch.bfloat16)
    cu = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
    cache = torch.tensor([3, 8, 13], dtype=torch.int32, device=DEV)
    has_false = torch.tensor([False, False, False], device=DEV)
    has_true = torch.tensor([True, True, True], device=DEV)
    report = {"lengths": lengths, "cases": []}

    for state_kind in ("zero", "dirty"):
        if state_kind == "zero":
            states = torch.zeros(16, WIDTH - 1, DIM).to(DEV, torch.bfloat16)
        else:
            states = (torch.randn(16, WIDTH - 1, DIM, generator=generator) * 0.1).to(
                DEV, torch.bfloat16
            )
        for init_name, has in (("false", has_false), ("true", has_true)):
            batch_states = states.clone()
            batch_out = conv(
                x, weight, batch_states,
                torch.tensor(cu, dtype=torch.int32, device=DEV), cache, has,
            )
            rows = torch.split(batch_out.transpose(0, 1), lengths, dim=0)
            isolated_diffs = []
            for index, (start, length) in enumerate(zip(cu[:-1], lengths)):
                ref_states = states.clone()
                ref_out = conv(
                    x[start : start + length], weight, ref_states,
                    torch.tensor([0, length], dtype=torch.int32, device=DEV),
                    cache[index : index + 1], has[index : index + 1],
                )
                ref_row = ref_out.transpose(0, 1)
                isolated_diffs.append(
                    float((rows[index].float() - ref_row.float()).abs().max())
                )
            record = {
                "state_cache": state_kind,
                "has_initial_state": init_name,
                "batch_vs_isolated_max_abs_diff_per_seq": isolated_diffs,
                "isolated_all_equal": max(isolated_diffs) == 0.0,
            }
            report["cases"].append(record)
            print(json.dumps(record, ensure_ascii=False))

    # The gather contract: False must return zeros without reading the cache.
    state = (torch.randn(16, 24, 128, 128, generator=generator) * 0.1).to(
        DEV, torch.bfloat16
    )
    indices = torch.tensor([3, 8, 13], dtype=torch.int64, device=DEV)
    for flag in (False, True):
        flags = torch.tensor([flag] * 3, device=DEV)
        gathered = gather_ssm_states(state, indices, flags, output_dtype=torch.bfloat16)
        expected = (
            state[indices] if flag
            else torch.zeros(3, 24, 128, 128).to(DEV, torch.bfloat16)
        )
        diff = float((gathered.float() - expected.float()).abs().max())
        record = {
            "gather_has_initial_state": flag,
            "max_abs_diff_vs_expected": diff,
            "all_equal": diff == 0.0,
        }
        report["cases"].append(record)
        print(json.dumps(record, ensure_ascii=False))

    report["fails"] = sum(1 for case in report["cases"] if not (case.get("all_equal") or case.get("isolated_all_equal")))
    print(json.dumps({"fails": report["fails"], "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
