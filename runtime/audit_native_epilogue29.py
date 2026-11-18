#!/usr/bin/env python3
"""Read-only audit of the fixed CANN 9.2.1 full MatMul/AR/AddRMSNorm API.

Run without loading ACL, torch, torch_npu, extensions, or any NPU runtime:
    ssh clef python3 - < runtime/audit_native_epilogue29.py > /tmp/epilogue29.json

Offsets below apply only to the exact SHA-256 checked before interpreting ELF.
This records static support evidence; it never claims a device test passed.
"""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
from datetime import datetime, timezone


SDK = Path('/usr/local/Ascend/cann-9.2.1')
LIB = SDK / 'aarch64-linux/lib64/libopapi.so'
EXPECTED_SHA = 'eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456'
OPS = ('MatmulAllReduceAddRmsNorm', 'InplaceMatmulAllReduceAddRmsNorm')


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'realpath': str(path.resolve()),
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def elf_reader(data):
    if data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 183:
        raise ValueError('Expected little-endian ELF64 AArch64')
    phoff = struct.unpack_from('<Q', data, 32)[0]
    entsize, count = struct.unpack_from('<HH', data, 54)
    segments = []
    for i in range(count):
        kind, _, offset, va, _, size, _, _ = struct.unpack_from('<IIQQQQQQ', data, phoff + i * entsize)
        if kind == 1:
            segments.append((va, size, offset))

    def read(va, size):
        for base, length, offset in segments:
            if base <= va and va + size <= base + length:
                return data[offset + va - base:offset + va - base + size]
        raise ValueError('Address outside file-backed PT_LOAD: ' + hex(va))
    return read


def disassemble(start, end):
    return subprocess.check_output(
        ['objdump', '-d', '-C', '--start-address=' + hex(start),
         '--stop-address=' + hex(end), str(LIB)], text=True)


def main():
    lib = identity(LIB)
    if lib['sha256'] != EXPECTED_SHA:
        raise SystemExit('Unknown library SHA; fixed offsets must be re-audited: ' + lib['sha256'])
    read = elf_reader(LIB.read_bytes())
    # Observed generated API: x2=&socNames[0], x3=1 at NnopbaseAddSocNameList.
    expected_words = {
        0x264c114: 0xb00044a0,  # adrp x0, 0x2ee1000
        0x264c118: 0x91030001,  # add x1, x0, #0xc0
        0x264c124: 0xd2800023,  # mov x3, #1
        0x264c128: 0xaa0103e2,  # mov x2, x1
        0x264c134: 0x976e9533,  # bl NnopbaseAddSocNameList@plt
    }
    for address, expected in expected_words.items():
        if struct.unpack('<I', read(address, 4))[0] != expected:
            raise ValueError('SoC-list instruction mismatch at ' + hex(address))
    name_pointer = struct.unpack('<Q', read(0x2ee10c0, 8))[0]
    soc = read(name_pointer, 64).split(b'\0', 1)[0].decode('ascii')
    if soc != 'ascend910b':
        raise ValueError('Unexpected SoC name: ' + soc)

    tbe = SDK / 'opp/built-in/op_impl/ai_core/tbe'
    records = []
    opinfo_count = 0
    for path in sorted((tbe / 'config').rglob('*.json')):
        opinfo_count += 1
        contents = json.loads(path.read_text())
        exact = [name for name in OPS if name in contents]
        if 'ascend910_93' in str(path) or exact:
            records.append({**identity(path), 'target_ops': exact})
    binaries = []
    for path in sorted((tbe / 'kernel/config/ascend910_93').rglob('binary_info_config.json')):
        contents = json.loads(path.read_text())
        binaries.append({**identity(path), 'target_ops': [name for name in OPS if name in contents]})
    if not records or not binaries:
        raise ValueError('Missing registry inventory; cannot classify package support')
    if any(record['target_ops'] for record in records + binaries):
        raise ValueError('Registry differs from audited package; re-evaluate support before closing')

    headers = []
    for stem in ('matmul_all_reduce_add_rms_norm', 'inplace_matmul_all_reduce_add_rms_norm'):
        path = SDK / ('aarch64-linux/include/aclnnop/aclnn_' + stem + '.h')
        headers.append({**identity(path), 'text': path.read_text()})
    source_dir = tbe / 'impl/ops_transformer/ascendc/matmul_all_reduce_add_rms_norm/op_kernel'
    sources = []
    for relative in ('arch22/matmul_all_reduce_add_rms_norm.cpp', 'mm_allreduce_add_rms_norm_910_general.h'):
        path = source_dir / relative
        sources.append({**identity(path), 'text': path.read_text()})
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(LIB)], text=True)
    result = {
        'collected_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'static_read_only_audit', 'device_execution': False,
        'library': lib, 'headers': headers,
        'symbols': [line for line in symbols.splitlines() if 'MatmulAllReduceAddRmsNorm' in line],
        'soc_list': {'api': 'aclnnInnerMatmulAllReduceAddRmsNormGetWorkspaceSize',
                     'call_va': '0x264c134', 'array_va': '0x2ee10c0',
                     'string_va': hex(name_pointer), 'count': 1, 'values': [soc],
                     'disassembly': disassemble(0x264c0f4, 0x264c1e0)},
        'routes': {
            'public_to_inner': disassemble(0x1e58e60, 0x1e58e88),
            'inner_to_generated': disassemble(0x1e5f510, 0x1e5f568)},
        'opinfo_files_scanned_all_socs': opinfo_count,
        'opinfo': records, 'a3_binary_indices': binaries, 'source_evidence': sources,
        'conclusion': 'Fixed library support list is ascend910b only; no A3 binding/device prescreen.',
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
