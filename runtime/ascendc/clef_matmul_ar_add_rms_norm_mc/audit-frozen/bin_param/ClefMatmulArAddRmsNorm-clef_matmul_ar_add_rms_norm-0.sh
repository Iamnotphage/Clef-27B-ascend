#!/bin/bash
echo "[ascend910_93] Generating ClefMatmulArAddRmsNorm_b87a0f8f165238ff7a5a2f429bc4f514 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(asc_opc $1 --main_func=clef_matmul_ar_add_rms_norm --input_param=/workspace/yangchen/work/wave3-e-audit-22/runtime/ascendc/clef_matmul_ar_add_rms_norm_mc/build_out/op_kernel/ClefMatmulArAddRmsNorm_ascend910_93/bin_param/ClefMatmulArAddRmsNorm_b87a0f8f165238ff7a5a2f429bc4f514_param.json --soc_version=Ascend910_9391                 --output=$2 --impl_mode=high_performance,optional --op_mode=dynamic  --simplified_key_mode=0)

echo "${res}"

if ! test -f $2/ClefMatmulArAddRmsNorm_b87a0f8f165238ff7a5a2f429bc4f514.json ; then
  echo "$2/ClefMatmulArAddRmsNorm_b87a0f8f165238ff7a5a2f429bc4f514.json not generated!"
  exit 1
fi

if ! test -f $2/ClefMatmulArAddRmsNorm_b87a0f8f165238ff7a5a2f429bc4f514.o ; then
  echo "$2/ClefMatmulArAddRmsNorm_b87a0f8f165238ff7a5a2f429bc4f514.o not generated!"
  exit 1
fi
echo "[ascend910_93] Generating ClefMatmulArAddRmsNorm_b87a0f8f165238ff7a5a2f429bc4f514 Done"
