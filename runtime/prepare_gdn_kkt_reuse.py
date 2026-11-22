#!/usr/bin/env python3
"""Prepare private, separately named stock/reuse GDN sources; never builds or runs.

Input must be the fixed upstream release. No installed wheel, shared GEV source,
service or device is changed. The public L2 adapter is omitted from these private
low-level diagnostic operators; production-wheel calibration remains required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

RELEASE = '5cc699069c7f91a35aabd583032c8720a0b626fa'
OP = Path('fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd')
EXPECTED = {
    'op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_tiling.cpp': 'd191e7ff90880f7c3b40db89e0763edf2703d565bbb9f00eaf0e4d0dae91b975',
    'op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp': '89f3360ad4afe86fcfef7f7d69d53ee9180350689f80e5918e2105daf7fecc5a',
    'op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22_struct.h': '96ad75ed0f1961d52d3ac6a97b7999d8252b41b37712be86c881b7e4491a9851',
    'op_kernel/internal/arch22/operators/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h': 'ddc8fc1f58cc712aae92811b68b38c17ff24bd9789579360af9b978489a403b8',
    'op_kernel/internal/arch22/operators/chunk_kkt_solve_tri/op_kernel/chunk_kkt_cube.h': '426394dd9c1ec82481c3c8dbfa34abf46a91ae4549e92518821716e88dd192d9',
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('unexpected release source: '+old[:80])
    return text.replace(old, new, 1)


def change(path, transform):
    original = path.read_text()
    result = transform(original)
    path.write_text(result)


def reuse_patch(root):
    kernel = root/'op_kernel/internal/arch22'
    structure = kernel/'chunk_gated_delta_rule_fwd_arch22_struct.h'
    change(structure, lambda s: replace_once(s,
        '    uint64_t frontWuWorkspaceOffset;',
        '    uint64_t frontWuWorkspaceOffset;\n'
        '    uint64_t kktReuse;  // Private ABI: raw score only, all ABC tasks stay Hv.'))
    host = root/'op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_tiling.cpp'
    change(host, lambda s: replace_once(s,
        '    abc.scoreWorkspaceBytes =\n'
        '        AlignUp(abc.taskNum * abc.BT * abc.BT * sizeof(float), WORKSPACE_ALIGNMENT);',
        '''    // The Python dispatcher additionally checks canonical host sequence metadata.
    // A single sequence is represented by cu[2] even for the S1 dense request.
    const auto* finalStateFlag = context->GetAttrs()->GetAttrPointer<bool>(0);
    trailer.kktReuse = finalStateFlag != nullptr && !*finalStateFlag &&
        useFp32Solve && isBf16 && rawGLayout == 1 &&
        qkvLayout == 1 && oLayout == 1 && batch == 1 && heads == 8 && valueHeads == 24 &&
        kDim == 128 && vDim == 128 && *chunkSize == 128 && tokens > 0 && tokens <= 1595 &&
        (!isVarlen || cuShape->GetStorageShape().GetDim(0) == 2) &&
        context->GetOptionalInputDesc(INPUT_GK) == nullptr &&
        context->GetOptionalInputDesc(INPUT_INITIAL_STATE) == nullptr ? 1 : 0;
    const uint64_t scoreTasks = trailer.kktReuse ? abc.B * abc.Hk * abc.NT : abc.taskNum;
    abc.scoreWorkspaceBytes =
        AlignUp(scoreTasks * abc.BT * abc.BT * sizeof(float), WORKSPACE_ALIGNMENT);'''))
    main = kernel/'chunk_gated_delta_rule_fwd_arch22.cpp'
    change(main, lambda s: replace_once(s,
        '    NsChunkScaledDotKkt::ChunkScaledDotKkt<InputT, InputT, true> kkt;',
        '    NsChunkScaledDotKkt::ChunkScaledDotKkt<InputT, InputT, true> kkt;\n'
        '    kkt.ConfigureSharedScore(phase6->kktReuse != 0);'))
    change(main, lambda s: replace_once(s,
        '        kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &abc);',
        '''        // Only the producer uses Hk tasks. Every physical group still initializes,
        // drains FIX_M, sends SCORE_READY once and joins the existing global barrier.
        auto producer = abc;
        if constexpr (kPreparedCumsum) {
            if (phase6->kktReuse != 0) {
                producer.taskNum = abc.B * abc.Hk * abc.NT;
                producer.tilesPerCore = (producer.taskNum + abc.usedAicNum - 1) / abc.usedAicNum;
            }
        }
        kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &producer);'''))
    consumer = kernel/'operators/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h'
    change(consumer, lambda s: replace_once(s, '    __aicore__ inline void Init(GM_ADDR k,',
        '    __aicore__ inline void ConfigureSharedScore(bool enabled) { sharedScore_ = enabled; }\n\n'
        '    __aicore__ inline void Init(GM_ADDR k,'))
    change(consumer, lambda s: replace_once(s,
        '        scoreGm.SetGlobalBuffer((__gm__ float *)scoreWorkspace, taskNum_ * BT_ * BT_);',
        '        const int64_t scoreTasks = sharedScore_ ? B_ * Hk_ * NT_ : taskNum_;\n'
        '        scoreGm.SetGlobalBuffer((__gm__ float *)scoreWorkspace, scoreTasks * BT_ * BT_);'))
    change(consumer, lambda s: replace_once(s,
        '        const int64_t scoreBaseOffset = task * BT_ * BT_;',
        '        const int64_t scoreTask = sharedScore_\n'
        '            ? (b * Hk_ + h / hvPerHk_) * NT_ + chunk : task;\n'
        '        const int64_t scoreBaseOffset = scoreTask * BT_ * BT_;'))
    change(consumer, lambda s: replace_once(s, '    int64_t taskHeads_ = 0;',
        '    int64_t taskHeads_ = 0;\n    bool sharedScore_ = false;'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not a.prepare:
        print(json.dumps(dict(status='plan', release=RELEASE,
            private_ops=['ClefGdnKktStock', 'ClefGdnKktReuse'], device_actions=False)))
        return
    if not a.source or not a.output or a.output.exists():
        p.error('prepare requires source and a NEW output directory')
    for rel, expected in EXPECTED.items():
        if hashlib.sha256((a.source/OP/rel).read_bytes()).hexdigest() != expected:
            p.error('release source SHA mismatch: '+rel)
    shutil.copytree(a.source, a.output)
    # Keep the diagnostic registry separate from the production wheel vendor.
    build_script = a.output/'build.sh'
    change(build_script, lambda s: s.replace('vendor_name="fla_npu"',
        'vendor_name="clef40_kkt"').replace('} | gawk ', '} | awk '))
    source = a.output/OP
    records = {}
    for mode, name in [('stock', 'ClefGdnKktStock'), ('reuse', 'ClefGdnKktReuse')]:
        snake = 'clef_gdn_kkt_'+mode
        target = source.parent/snake
        shutil.copytree(source, target)
        # Produce an SDK-generated L0 two-stage ABI with a unique operator type.
        # The original L2 adapter is retained in the source archive for comparison.
        shutil.rmtree(target/'op_host/op_api')
        cmake = (target/'op_host/CMakeLists.txt').read_text()
        cmake = cmake[cmake.index('if (BUILD_OPEN_PROJECT)'):]
        cmake = cmake.replace('ACLNNTYPE aclnn_exclude', 'ACLNNTYPE aclnn')
        cmake = cmake.replace('target_sources(op_host_aclnnExc PRIVATE',
                              'target_sources(op_host_aclnn PRIVATE')
        (target/'op_host/CMakeLists.txt').write_text(cmake)
        for path in target.rglob('*'):
            if not path.is_file() or path.suffix not in ('.cpp', '.h', '.txt'):
                continue
            content = path.read_text()
            # Host tiling entrypoints and private struct names share this prefix.
            # Rename them too so stock/reuse can coexist in one host DSO.
            content = content.replace('ChunkGatedDeltaRuleFwd', name)
            content = re.sub(r'\bchunk_gated_delta_rule_fwd\b', snake, content)
            path.write_text(content)
        entry = target/'op_kernel/chunk_gated_delta_rule_fwd.cpp'
        entry.rename(entry.with_name(snake+'.cpp'))
        definition = target/'op_host/chunk_gated_delta_rule_fwd_def.cpp'
        definition.rename(definition.with_name(snake+'_def.cpp'))
        change(target/'op_host/CMakeLists.txt', lambda s: s.replace(
            'chunk_gated_delta_rule_fwd_def.cpp', snake+'_def.cpp'))
        if mode == 'reuse':
            reuse_patch(target)
        host = target/'op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_tiling.cpp'
        enabled = 'trailer.kktReuse' if mode == 'reuse' else '0ULL'
        def observe(s):
            s = '#include <cstdio>\n#include <cstdlib>\n'+s
            return replace_once(s, '    context->SetScheduleMode(1);',
                '    context->SetScheduleMode(1);\n'
                '    if (const char* path = std::getenv("CLEF_GDN_KKT_META")) {\n'
                '        if (FILE* file = std::fopen(path, "a")) {\n'
                '            std::fprintf(file, "'+name+' key=%lu reuse=%lu B=%lu Hk=%lu Hv=%lu T=%lu BT=%lu NT=%lu tasks=%lu scoreBytes=%lu scoreOffset=%lu solveInputOffset=%lu sysBytes=%lu\\n",\n'
                '                context->GetTilingKey(), static_cast<uint64_t>('+enabled+'), abc.B, abc.Hk, abc.Hv, abc.T, abc.BT, abc.NT, abc.taskNum,\n'
                '                abc.scoreWorkspaceBytes, trailer.scoreWorkspaceOffset, trailer.solveFp32InputOffset, systemWorkspace);\n'
                '            std::fclose(file);\n'
                '        }\n'
                '    }')
        change(host, observe)
        records[mode] = {str(path.relative_to(a.output)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in target.rglob('*') if path.is_file()}
    (a.output/'clef-kkt-preparation.json').write_text(json.dumps(dict(
        release=RELEASE, status='private_source_prepared_uncompiled', source_sha256=records), indent=2)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
