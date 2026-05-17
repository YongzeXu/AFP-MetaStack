#!/usr/bin/env bash
# 官方 train/test：用 Focal 重训 prepare 中的 MLP；**默认三路 NPZ**（ESM2 + SaProt 头 + GearNet，`p_esm`/`p_saprot`/`p_gearnet`），
# 写出 val_probs_for_ensemble.npz + test_probs_for_ensemble.npz，供 Meta 与「Focal 对齐」的独立集评估共用。
# 默认 γ=1.5、目录 `..._focal_g15`（与两路实验 `..._focal_g15_esm_gear` 区分）。若只要两路：对 prepare 加 `--npz_two_branch_esm_gear` 并设 FOCAL_OFFICIAL_OUT_DIR 指向 `_esm_gear` 目录。
#
# 用法：
#   bash classifier_scripts/run_prepare_focal_val_test_official.sh
# 可选环境变量：
#   FOCAL_GAMMA=1.5
#   FOCAL_OFFICIAL_OUT_DIR=/path/to/out   （覆盖默认目录名）
#   SKIP_IF_EXISTS=1                       （若 test_npz 已存在则跳过 prepare）
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
else
  PY="python3"
fi
cd "$ROOT"

GAMMA="${FOCAL_GAMMA:-1.5}"
GAMMA_TAG="${GAMMA//./p}"
OUT_DIR="${FOCAL_OFFICIAL_OUT_DIR:-${ROOT}/trained/ensemble_prep_weighted_heads_gear_npz_focal_g${GAMMA_TAG}}"
mkdir -p "$OUT_DIR"

TEST_NPZ="${OUT_DIR}/test_probs_for_ensemble.npz"
if [[ "${SKIP_IF_EXISTS:-0}" == "1" && -f "$TEST_NPZ" ]]; then
  echo ">>> 已存在 $TEST_NPZ 且 SKIP_IF_EXISTS=1，跳过 prepare。"
  echo ">>> val_npz: ${OUT_DIR}/val_probs_for_ensemble.npz"
  exit 0
fi

echo ">>> Focal 官方 val+test prepare -> $OUT_DIR（gamma=$GAMMA，耗时较长，建议 nohup/tmux）"
"$PY" classifier_scripts/prepare_ensemble_probs_afp_e.py \
  --gearnet_branch weighted_heads \
  --esm2_cache_dir "${ROOT}/features/afp_e_esm2_saprot_token_bins" \
  --gearnet_emb_cache_dir "${ROOT}/features/afp_e_gearnet_mc_emb_cache" \
  --out_dir "$OUT_DIR" \
  --phase both \
  --loss focal \
  --focal_gamma "$GAMMA"

echo ">>> 完成。请将独立评估的 VAL_NPZ 指到："
echo "    ${OUT_DIR}/val_probs_for_ensemble.npz"
echo ">>> 并与独立集 prepare 使用相同 --loss focal --focal_gamma $GAMMA（三路默认含 SaProt；见 run_focal_lift_independent_pipeline.sh）。"
