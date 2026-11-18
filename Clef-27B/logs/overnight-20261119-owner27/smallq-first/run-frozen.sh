#!/usr/bin/env bash
set -euo pipefail
exec /workspace/yangchen/venvs/clef-py312/bin/python /workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py \
  --run-dir /workspace/yangchen/logs/joint-smallq-27-2e8af5d-first \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-5c045eb \
  --tree /workspace/yangchen/work/src/vllm \
  --python /workspace/yangchen/venvs/clef-py312/bin/python \
  --activate /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh \
  --launch /workspace/yangchen/logs/perf-wave2-20261118/launch.sh \
  --wait /workspace/yangchen/logs/perf-wave2-20261118/wait.sh \
  --device 0 --port 18143 --candidate-kind head-postimage \
  --stop-script /tmp/d0cc/d0_stop.py \
  --expect-stop-sha d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886 \
  --postimage-rel runtime/joint_smallq/clef_joint_schema.py \
  --postimage-source /tmp/head_89eb/runtime/joint_smallq/clef_joint_schema.py \
  --postimage-sha cf02419978ce62f18ca908b8d4c9646531fb00d3b07871b310576f963e5ec612 \
  --installer-rel runtime/apply_joint_smallq.py \
  --installer-source /tmp/head_89eb/runtime/apply_joint_smallq.py \
  --installer-sha 47d34e07d468dfbfce93f7ac8d4eeff73c847b83fd56df6a167591a3b5b212c6 \
  --probe-rel runtime/joint_smallq_probe.py \
  --probe-source /tmp/head_89eb/runtime/joint_smallq_probe.py \
  --probe-sha 4d9d7736a35598b23badd913c2797f15e988aeeecca1f13352ea07f549ea1478 \
  --payload /tmp/head_89eb/runtime/joint_smallq/clef_smallq_kernel.py:runtime/joint_smallq/clef_smallq_kernel.py:08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff \
  --client-rel '' --env-flag CLEF_JOINT_SMALLQ --probe-schema joint-smallq-probe/1 \
  --preimage-sha f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863 \
  --stop-timeout 120 --probe-timeout 900 --wait-timeout 900
