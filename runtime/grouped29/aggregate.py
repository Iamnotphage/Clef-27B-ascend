"""CPU reconstruction of raw output gates and complete two-rank timing evidence."""
import json
import math
import os
from pathlib import Path
from contract import OWNER, N1_CELL, N1_GROUP, SDK, check_sha, load_gate, n1_inputs, sha256
from window_common import CELL, PARAMETERS
from timing_contract import aggregate_samples


def load_raw(root, name, spec, shape, dtype):
    import torch
    sizes = {'torch.bfloat16': 2, 'torch.float32': 4}
    if (set(spec) != {'shape', 'dtype', 'bytes', 'sha256'} or
            spec['shape'] != shape or spec['dtype'] != dtype or
            spec['bytes'] != math.prod(shape) * sizes[dtype]):
        raise ValueError('Raw tensor descriptor mismatch: ' + name)
    path = root / (name + '.bin')
    if path.stat().st_size != spec['bytes']:
        raise ValueError('Raw tensor size mismatch: ' + name)
    check_sha(path, spec['sha256'])
    tensor = torch.frombuffer(bytearray(path.read_bytes()), dtype=getattr(torch, dtype.split('.')[1])).reshape(shape)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError('Raw tensor contains nonfinite values: ' + name)
    return tensor


def aggregate(cell, stage, identity, pgid):
    # CPU aggregate is a separate process from torchrun; disable backend autoload.
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    torch.set_num_threads(1)
    evaluate = load_gate(stage / 'fixed_gate.py')
    metas, records, rechecks = [], [], []
    provider = ('prepare/launch=' + str((SDK / 'lib64/libopapi.so').resolve()) +
                ';metadata=' + str((SDK / 'lib64/libnnopbase.so').resolve()))
    for rank in (0, 1):
        root = cell / f'rank{rank}'
        meta = json.loads((root / 'result.json').read_text())
        expected = {**CELL, 'schema': 'grouped29-probe/1', 'rank': rank, 'owner': OWNER,
                    'stage': 'complete', 'identity_sha256': sha256(stage / 'identity.json'),
                    'identity': identity, 'parameters': PARAMETERS, 'pgid': pgid,
                    'n1_cell': str(N1_CELL), 'provider': provider,
                    'pass': True, 'performance_valid': True, 'n1_input_match': True,
                    'inputs_unchanged': True, 'inputs_unchanged_after': True,
                    'output_preconditions': True, 'group': N1_GROUP,
                    'native_grouped_rstd': None,
                    'rstd_source': 'actual output of subsequent native AddRMSNorm'}
        if any(meta.get(k) != v for k, v in expected.items()) or meta.get('error') is not None:
            raise ValueError(f'rank{rank} result identity/status mismatch')
        if meta.get('inputs') != n1_inputs(rank):
            raise ValueError(f'rank{rank} N1 input hash mismatch')
        if (meta.get('layout', {}).get('candidate_weight_format') != 2 or
                meta.get('layout', {}).get('service_weight_format') != 29 or
                not isinstance(meta.get('group'), str) or not meta['group']):
            raise ValueError(f'rank{rank} ND/NZ/group mismatch')
        for name in ('candidate_weight_cpu_layout_wall_ms', 'candidate_weight_upload_wall_ms', 'service_nz_setup_wall_ms'):
            value = meta['layout'].get(name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('Missing/nonfinite initialization cost: ' + name)
        for name in ('x', 'r', 'gamma'):
            load_raw(root, name, meta['inputs'][name], meta['inputs'][name]['shape'], 'torch.bfloat16')
        arms_by_prefix = {'': ('baseline', 'nd_native', 'candidate'),
                          **{f'round{i}.': ('baseline', 'candidate') for i in range(4)},
                          'profile.': ('baseline', 'candidate')}
        expected_files = {prefix + name + '.' + arm for prefix, arms in arms_by_prefix.items()
                          for arm in arms for name in ('y', 'residual', 'rstd', 'reduced')}
        if set(meta.get('files', {})) != expected_files:
            raise ValueError(f'rank{rank} incomplete raw output set')
        first_baseline = None
        rank_checks = []
        for prefix, arms in arms_by_prefix.items():
            data = {}
            for arm in arms:
                data[arm] = {}
                for name in ('y', 'residual', 'rstd', 'reduced'):
                    key = prefix + name + '.' + arm
                    shape, dtype = ([CELL['rows'], 1], 'torch.float32') if name == 'rstd' else ([CELL['rows'], 5120], 'torch.bfloat16')
                    data[arm][name] = load_raw(root, key, meta['files'][key], shape, dtype)
            gate = evaluate(data['candidate'], data['baseline'])
            if not gate['pass']:
                raise ValueError(f'rank{rank} raw fixed gate failed at {prefix!r}')
            if not prefix:
                first_baseline = data['baseline']
                if gate != meta.get('fixed_gate'):
                    raise ValueError(f'rank{rank} claimed initial gate differs from raw gate')
                nd = evaluate(data['nd_native'], data['baseline'])
                if nd != meta.get('nd_diagnostic_gate'):
                    raise ValueError('ND diagnostic differs from raw output')
                rank_checks.append({'prefix': prefix, 'fixed_gate': gate, 'nd_diagnostic': nd})
            else:
                stable = evaluate(data['baseline'], first_baseline)
                if not stable['pass']:
                    raise ValueError(f'rank{rank} baseline drift at {prefix!r}')
                if prefix == 'profile.':
                    if gate != meta.get('profile_gate'):
                        raise ValueError('Profile gate differs from raw output')
                else:
                    index = int(prefix[5:-1])
                    rounds = meta.get('timing', {}).get('rounds', [])
                    if len(rounds) != 4 or rounds[index] != {'round': index, 'gate': gate, 'baseline_stable': stable, 'pass': True}:
                        raise ValueError('Claimed timing gate differs from raw output')
                rank_checks.append({'prefix': prefix, 'fixed_gate': gate, 'baseline_stable': stable})
        profile = root / 'profile'
        files = meta.get('profile_files', {})
        if not files or set(files) != {str(p.relative_to(profile)) for p in profile.rglob('*') if p.is_file()}:
            raise ValueError('Incomplete profiler evidence')
        for name, spec in files.items():
            path = (profile / name).resolve()
            if not path.is_relative_to(profile.resolve()) or path.stat().st_size != spec['bytes']:
                raise ValueError('Profiler file identity mismatch')
            check_sha(path, spec['sha256'])
        check_sha(root / 'samples.jsonl', meta['samples_sha256'])
        records.append([json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()])
        metas.append(meta)
        rechecks.append({'rank': rank, 'result_sha256': sha256(root / 'result.json'),
                         'samples_sha256': meta['samples_sha256'], 'raw_gates': rank_checks})
    for name in ('x', 'r', 'gamma'):
        if metas[0]['inputs'][name] != metas[1]['inputs'][name]:
            raise ValueError('Cross-rank shared input mismatch: ' + name)
    for rank in (0, 1):
        if metas[rank].get('both_rank_inputs') != [m['inputs'] for m in metas]:
            raise ValueError('Cross-rank input exchange mismatch')
    summary = aggregate_samples(records, metas)
    summary.update({'rank_evidence': rechecks, 'cell': CELL, 'parameters': PARAMETERS,
                    'comm_groups': [m['group'] for m in metas],
                    'boundary': 'GroupedMatMulAllReduce + separate native AddRMSNorm',
                    'first_call_wall_ms': [m['first_call_wall_ms'] for m in metas],
                    'no_device_fusion_claim': True})
    (cell / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return {'summary_path': str(cell / 'summary.json'), 'summary_sha256': sha256(cell / 'summary.json'),
            'measure': summary['measure'], 'rank_evidence': rechecks}
