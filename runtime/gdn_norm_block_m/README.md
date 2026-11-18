# GDN output-norm `BLOCK_M` preimage

`layernorm_gated_preimage.py` is the exact upstream file that
`runtime/apply_gdn_norm_block_m.py` hash-gates on:

- source: `vllm_ascend/ops/triton/layernorm_gated.py`
- tree: vLLM-Ascend `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`
- copied: 2026-11-18, read-only from
  `/workspace/yangchen/work/src/vllm-ascend`
- preimage SHA-256: `68f61d7f13547a11ca38eb33c0cd560dc6a569ad97c7fbc6681d1370db70a774`
- patched SHA-256: `a1093e1290afd48b546830a9ae0639f945910f44f9e22f12111b67aa72f8fe6c`

The file keeps its upstream header. It is never imported or executed here; the
installer and its offline test read it as text only, so no torch, torch_npu or
vllm_ascend import is involved in the CPU preparation.
