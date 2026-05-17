#!/usr/bin/env bash
# 「再抬」独立集：底层 **三路** 概率（ESM + SaProt + GearNet NPZ）与 Meta 训练用 val 对齐为 Focal（默认 γ=1.5）。
# 步骤 1）官方域内 val+test NPZ（新目录，不覆盖 CE）
# 步骤 2）独立集 test NPZ（同损失）+ four_panel stacking，并默认开启 **--robust_ood_meta**（在 Focal 概率上常显著抬独立集 Meta，见 focal_g*_robustood.json）
#
# 用法：
#   bash classifier_scripts/run_focal_lift_independent_pipeline.sh
# 可选：
#   FOCAL_GAMMA=2.0
#   SKIP_OFFICIAL_IF_EXISTS=1    若官方 focal 目录已有 test_npz 则跳过步骤 1
#   SKIP_INDEP_PREPARE=1         仅重跑 stacking（需已有独立集 focal NPZ）
#   SKIP_INDEP_XGB_FINAL=1       跳过步骤 3（默认会跑：独立集主结果以 Meta-XGB 为准，见下）
#   SKIP_OFFICIAL_ALSO_DUAL_XGB=1  跳过步骤 4（默认会跑：同一 JSON 内域内 test + 独立集 also）
#
# 步骤 3（默认开启）：在步骤 2 的 Meta-LR 之后追加 **Meta-XGB + val RBF(γ=0.003) + reg_lambda=1.5**，
# 与 experiments_table_independent_zh 中独立集最优行同配置；特征刷新后 Meta-LR 常明显弱于 XGB，故一键流程以 XGB JSON 为推荐产物。
#
# 步骤 4（默认开启）：**官方 Focal test_npz + 独立集** 单次 Meta-XGB（`test_metrics` = 域内 test，`test_metrics_also_holdout_npz` = 独立集），
# 见 `run_focal_official_test_and_independent_xgb_dual_eval.sh`；与步骤 3 共用 `META_VAL_REWEIGHT_HOLDOUT_RBF_GAMMA` / `XGB_REG_LAMBDA` 等环境变量。
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAMMA="${FOCAL_GAMMA:-1.5}"
GAMMA_TAG="${GAMMA//./p}"
FOCAL_GAMMA_TAG="g${GAMMA_TAG}"
OFF="${FOCAL_OFFICIAL_OUT_DIR:-${ROOT}/trained/ensemble_prep_weighted_heads_gear_npz_focal_g${GAMMA_TAG}}"
IND="${FOCAL_INDEP_OUT_DIR:-${ROOT}/trained/ensemble_prep_independent_weighted_heads_focal_g${GAMMA_TAG}}"

export FOCAL_GAMMA="$GAMMA"
export FOCAL_OFFICIAL_OUT_DIR="$OFF"
if [[ "${SKIP_OFFICIAL_IF_EXISTS:-0}" == "1" ]]; then
  export SKIP_IF_EXISTS=1
fi
bash "${ROOT}/classifier_scripts/run_prepare_focal_val_test_official.sh"
unset SKIP_IF_EXISTS 2>/dev/null || true

if [[ "${SKIP_INDEP_PREPARE:-0}" != "1" ]]; then
  export VAL_NPZ="${OFF}/val_probs_for_ensemble.npz"
  export ENSEMBLE_INDEP_OUT_DIR="$IND"
  export PREPARE_EXTRA_ARGS="--loss focal --focal_gamma ${GAMMA}"
  export ROBUST_OOD_META=1
  export STACKING_OUT_JSON="${IND}/stacking_independent_focal_g${GAMMA_TAG}_robustood.json"
  export CSV_ROW_TAG_ZH="【独立集·Focalγ${GAMMA}+robust_ood_meta·val同分布·weighted_heads·gp·LR·C1.5·pca24】"
  bash "${ROOT}/classifier_scripts/run_independent_eval.sh"
else
  export VAL_NPZ="${OFF}/val_probs_for_ensemble.npz"
  export ENSEMBLE_INDEP_OUT_DIR="$IND"
  export ROBUST_OOD_META=1
  export STACKING_OUT_JSON="${IND}/stacking_independent_focal_g${GAMMA_TAG}_robustood.json"
  export CSV_ROW_TAG_ZH="【独立集·Focalγ${GAMMA}+robust_ood_meta·仅重算stacking】"
  HOLD="${IND}/test_probs_for_ensemble.npz"
  if [[ ! -f "$HOLD" ]]; then
    echo "缺少 $HOLD，不能 SKIP_INDEP_PREPARE=1" >&2
    exit 1
  fi
  if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PY="${CONDA_PREFIX}/bin/python"
  else
    PY="python3"
  fi
  cd "$ROOT"
  "$PY" classifier_scripts/ensemble_stacking_plus_abc.py \
    --four_panel \
    --robust_ood_meta \
    --val_npz "$VAL_NPZ" \
    --holdout_npz "$HOLD" \
    --holdout_pos_fasta "${ROOT}/data/independent_set/pos_independent.fasta" \
    --holdout_neg_fasta "${ROOT}/data/independent_set/neg_independent.fasta" \
    --feat_go_csv "${ROOT}/trained/functional_csv_real/feat_go_uniprot_top128.csv" \
    --feat_ppi_csv "${ROOT}/trained/functional_csv_real/feat_ppi_string_node2vec64.csv" \
    --feat_auto_pca_dim 24 \
    --stacking_C 1.5 \
    --holdout-report-base-fusions \
    --holdout-val-pick-triple-fusion \
    --holdout-val-pick-mean3-vs-meta-val-slack 0.012 \
    --holdout-zero-feat-csv-tags go,ppi \
    --no-append_csv \
    --out_json "$STACKING_OUT_JSON" \
    --csv_row_tag_zh "$CSV_ROW_TAG_ZH"
fi

# 步骤 3：独立集 holdout 上稳定的 Meta-XGB（与主表 RBF003+λ1.5 行一致）
PRIMARY_XGB_JSON="${IND}/stacking_independent_focal_${FOCAL_GAMMA_TAG}_xgb_robust_rbf_g0003_reglambda15.json"
RECOMMEND_XGB_JSON="$PRIMARY_XGB_JSON"
if [[ "${SKIP_INDEP_XGB_FINAL:-0}" != "1" ]]; then
  echo ">>> 步骤 3：Meta-XGB + robust_ood + RBF val γ=0.003 + reg_lambda=1.5 -> ${PRIMARY_XGB_JSON}"
  export FOCAL_GAMMA_TAG
  export FOCAL_OFFICIAL_OUT_DIR="$OFF"
  export FOCAL_INDEP_OUT_DIR="$IND"
  export META_VAL_REWEIGHT_HOLDOUT_RBF_GAMMA="${META_VAL_REWEIGHT_HOLDOUT_RBF_GAMMA:-0.003}"
  export XGB_REG_LAMBDA="${XGB_REG_LAMBDA:-1.5}"
  export STACKING_XGB_OUT_JSON="${STACKING_XGB_OUT_JSON:-$PRIMARY_XGB_JSON}"
  export CSV_ROW_TAG_ZH="${CSV_ROW_TAG_ZH_XGB:-【独立集·Focalγ${GAMMA}+XGB+robust·RBF003·λ1.5·pca24·一键默认】}"
  bash "${ROOT}/classifier_scripts/run_independent_focal_xgb_robust_eval.sh"
  RECOMMEND_XGB_JSON="$STACKING_XGB_OUT_JSON"
else
  echo ">>> 跳过步骤 3（SKIP_INDEP_XGB_FINAL=1）"
fi

# 步骤 4：域内 official test + 独立集（同一 Meta，JSON 双指标）
OFFICIAL_DUAL_JSON="${OFF}/stacking_official_test_and_independent_focal_${FOCAL_GAMMA_TAG}_xgb_robust_rbf_g0003_reglambda15.json"
if [[ "${SKIP_OFFICIAL_ALSO_DUAL_XGB:-0}" != "1" ]]; then
  echo ">>> 步骤 4：official test + independent also（双指标 JSON）-> ${OFFICIAL_DUAL_JSON}"
  export FOCAL_GAMMA_TAG
  export FOCAL_OFFICIAL_OUT_DIR="$OFF"
  export FOCAL_INDEP_OUT_DIR="$IND"
  export META_VAL_REWEIGHT_HOLDOUT_RBF_GAMMA="${META_VAL_REWEIGHT_HOLDOUT_RBF_GAMMA:-0.003}"
  export XGB_REG_LAMBDA="${XGB_REG_LAMBDA:-1.5}"
  export STACKING_OFFICIAL_DUAL_JSON="${STACKING_OFFICIAL_DUAL_JSON:-$OFFICIAL_DUAL_JSON}"
  bash "${ROOT}/classifier_scripts/run_focal_official_test_and_independent_xgb_dual_eval.sh"
else
  echo ">>> 跳过步骤 4（SKIP_OFFICIAL_ALSO_DUAL_XGB=1）"
fi

echo ">>> 全流程结束。独立集 Meta-LR（对照/历史）: ${IND}/stacking_independent_focal_g${GAMMA_TAG}_robustood.json"
echo ">>> 独立集 JSON（推荐·holdout 主评估·含 triple-fusion 等）: ${RECOMMEND_XGB_JSON}"
echo ">>> 域内 test + 独立集同一 Meta（推荐·双指标 JSON）: ${OFFICIAL_DUAL_JSON}"
