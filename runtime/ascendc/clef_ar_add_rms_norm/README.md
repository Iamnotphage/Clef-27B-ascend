# ClefArAddRmsNorm (gate-B scaffold)

Official `npu_op_package` scaffold for the T1 fusion study.  The current
`op_kernel` is a byte-copy stub: it does **no** communication and **no**
normalization.  Its only job is to let the official build resolve the MC2
tiling API (`AscendC::Mc2CcTilingConfig`), which cannot link in a standalone
g++ program because the op_host libraries are built with
`-fvisibility=hidden`.

The gate-C dual-rank test already proved that `HcclCreateOpResCtx` returns a
non-null context with `AllReduce=level0:{ring|doublering|fullmesh}` on
910_93, while an empty `algConfig` is rejected at parse time.

Build (remote, after `source scripts/activate_env.sh`):

```bash
cd runtime/ascendc/clef_ar_add_rms_norm && bash build.sh
```

Do not install this vendor into the serving `ASCEND_CUSTOM_OPP_PATH` until the
kernel actually implements AllReduce+Add+RMSNorm.
