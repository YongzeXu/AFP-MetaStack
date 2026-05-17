#!/usr/bin/env bash
# 单次 Meta-XGB + robust_ood：**主表 test_metrics = 官方 Focal test_npz**；**test_metrics_also_holdout_npz = 独立集（纯 Meta）**；
# **test_metrics_also_extrapolation_val_picked** = 独立集上按 val 与 `run_independent_focal_xgb_robust_eval` 相同 triple 规则选 Meta/mean3/median3 后的指标。
# 不重训 prepare；需已有 OFF/{val,test}_probs_for_ensemble.npz 与 IND/test_probs_for_ensemble.npz。
#
# 用法：
#   bash classifier_scripts/run_focal_official_test_and_independent_xgb_dual_eval.sh
# 环境变量与 run_independent_focal_xgb_robust_eval.sh 对齐：FOCAL_GAMMA_TAG、FOCAL_OFFICIAL_OUT_DIR、FOCAL_INDEP_OUT_DIR、
#   FEAT_AUTO_PCA_DIM、SCALE_META、META_CROSS_FEATURES、TRAIN_OOF_NPZ、META_TRAIN_OOF_AUGMENT_ROW_WEIGHT、
#   META_VAL_REWEIGHT_HOLDOUT_RBF_GAMMA（默认 0.003）、XGB_REG_LAMBDA（默认 1.5）、各 XGB_* 等。
# 输出路径：STACKING_OFFICIAL_DUAL_JSON（默认 OFF/stacking_official_test_and_independent_focal_${FOCAL_GAMMA_TAG}_xgb_robust_rbf_g0003_reglambda15.json）
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
GAMMA_TAG="${FOCAL_GAMMA_TAG:-g1p5}"
OFF="${FOCAL_OFFICIAL_OUT_DIR:-${ROOT}/trained/ensemble_prep_weighted_heads_gear_npz_focal_${GAMMA_TAG}}"
IND="${FOCAL_INDEP_OUT_DIR:-${ROOT}/trained/ensemble_prep_independent_weighted_heads_focal_${GAMMA_TAG}}"
VAL_NPZ="${OFF}/val_probs_for_ensemble.npz"
TEST_NPZ="${OFF}/test_probs_for_ensemble.npz"
IND_NPZ="${IND}/test_probs_for_ensemble.npz"
OUT_JSON="${STACKING_OFFICIAL_DUAL_JSON:-${OFF}/stacking_official_test_and_independent_focal_${GAMMA_TAG}_xgb_robust_rbf_g0003_reglambda15.json}"
FEAT_PCA_DIM="${FEAT_AUTO_PCA_DIM:-24}"
SCALE_META_FLAG=()
if [[ "${SCALE_META:-1}" != "0" ]]; then
  SCALE_META_FLAG=(--scale_meta)
fi
XGB_EXTRA=()
: "${XGB_REG_LAMBDA:=1.5}"
[[ -n "${XGB_N_ESTIMATORS:-}" ]] && XGB_EXTRA+=(--xgb-n-estimators "${XGB_N_ESTIMATORS}")
[[ -n "${XGB_MAX_DEPTH:-}" ]] && XGB_EXTRA+=(--xgb-max-depth "${XGB_MAX_DEPTH}")
[[ -n "${XGB_LEARNING_RATE:-}" ]] && XGB_EXTRA+=(--xgb-learning-rate "${XGB_LEARNING_RATE}")
[[ -n "${XGB_REG_LAMBDA:-}" ]] && XGB_EXTRA+=(--xgb-reg-lambda "${XGB_REG_LAMBDA}")
[[ -n "${XGB_MIN_CHILD_WEIGHT:-}" ]] && XGB_EXTRA+=(--xgb-min-child-weight "${XGB_MIN_CHILD_WEIGHT}")
[[ -n "${XGB_SUBSAMPLE:-}" ]] && XGB_EXTRA+=(--xgb-subsample "${XGB_SUBSAMPLE}")
[[ -n "${XGB_COLSAMPLE_BYTREE:-}" ]] && XGB_EXTRA+=(--xgb-colsample-bytree "${XGB_COLSAMPLE_BYTREE}")
[[ -n "${META_XGB_MISSING_ALPHA:-}" ]] && XGB_EXTRA+=(--meta-xgb-sample-weight-missing-alpha "${META_XGB_MISSING_ALPHA}")
[[ -n "${META_XGB_BOOTSTRAP_K:-}" ]] && XGB_EXTRA+=(--meta-xgb-bootstrap-k "${META_XGB_BOOTSTRAP_K}")
[[ -n "${META_XGB_BOOTSTRAP_SEED:-}" ]] && XGB_EXTRA+=(--meta-xgb-bootstrap-seed "${META_XGB_BOOTSTRAP_SEED}")
[[ -n "${META_XGB_SEED_ENSEMBLE_K:-}" ]] && XGB_EXTRA+=(--meta-xgb-seed-ensemble-k "${META_XGB_SEED_ENSEMBLE_K}")
G_RBF="${META_VAL_REWEIGHT_HOLDOUT_RBF_GAMMA:-0.003}"
XGB_EXTRA+=(--meta-val-reweight-holdout-rbf-gamma "${G_RBF}")
[[ -n "${META_VAL_REWEIGHT_HOLDOUT_ALPHA:-}" ]] && XGB_EXTRA+=(--meta-val-reweight-holdout-alpha "${META_VAL_REWEIGHT_HOLDOUT_ALPHA}")
[[ -n "${META_VAL_REWEIGHT_HOLDOUT_MAX_ROWS:-}" ]] && XGB_EXTRA+=(--meta-val-reweight-holdout-max-rows "${META_VAL_REWEIGHT_HOLDOUT_MAX_ROWS}")
[[ -n "${META_VAL_REWEIGHT_HOLDOUT_RBF_AVG_SEEDS:-}" ]] && XGB_EXTRA+=(--meta-val-reweight-holdout-rbf-avg-seeds "${META_VAL_REWEIGHT_HOLDOUT_RBF_AVG_SEEDS}")
[[ -n "${XGB_SCALE_POS_WEIGHT:-}" ]] && XGB_EXTRA+=(--xgb-scale-pos-weight "${XGB_SCALE_POS_WEIGHT}")
# test_npz 模式下 RBF 须指定参考集（与域内 stacking 族一致）
XGB_EXTRA+=(
  --meta-val-rbf-reference-holdout-npz "${IND_NPZ}"
  --meta-val-rbf-reference-pos-fasta "${ROOT}/data/independent_set/pos_independent.fasta"
  --meta-val-rbf-reference-neg-fasta "${ROOT}/data/independent_set/neg_independent.fasta"
)
META_CROSS_EXTRA=()
if [[ "${META_CROSS_FEATURES:-0}" == "1" ]]; then
  META_CROSS_EXTRA=(--meta_cross_features)
fi
TRAIN_OOF_EXTRA=()
if [[ -n "${TRAIN_OOF_NPZ:-}" ]]; then
  if [[ ! -f "${TRAIN_OOF_NPZ}" ]]; then
    echo "TRAIN_OOF_NPZ 指向的文件不存在: ${TRAIN_OOF_NPZ}" >&2
    exit 1
  fi
  TRAIN_OOF_EXTRA=(--train_oof_npz "${TRAIN_OOF_NPZ}")
fi
if [[ -n "${TRAIN_OOF_NPZ:-}" && -n "${META_TRAIN_OOF_AUGMENT_ROW_WEIGHT:-}" ]]; then
  XGB_EXTRA+=(--meta-train-oof-augment-row-weight "${META_TRAIN_OOF_AUGMENT_ROW_WEIGHT}")
fi

for f in "$VAL_NPZ" "$TEST_NPZ" "$IND_NPZ"; do
  if [[ ! -f "$f" ]]; then
    echo "缺少 $f（需先完成官方 Focal prepare 与独立集 prepare）" >&2
    exit 1
  fi
done

echo ">>> official test + independent also -> $OUT_JSON (feat_pca_dim=${FEAT_PCA_DIM}, RBF_gamma=${G_RBF}, SCALE_META=${SCALE_META:-1})"
"$PY" classifier_scripts/ensemble_stacking_plus_abc.py \
  --four_panel \
  --robust_ood_meta \
  --meta_classifier xgboost \
  "${SCALE_META_FLAG[@]}" \
  --val_npz "$VAL_NPZ" \
  --test_npz "$TEST_NPZ" \
  --also-holdout-npz "$IND_NPZ" \
  --also-holdout-pos-fasta "${ROOT}/data/independent_set/pos_independent.fasta" \
  --also-holdout-neg-fasta "${ROOT}/data/independent_set/neg_independent.fasta" \
  --feat_go_csv "${ROOT}/trained/functional_csv_real/feat_go_uniprot_top128.csv" \
  --feat_ppi_csv "${ROOT}/trained/functional_csv_real/feat_ppi_string_node2vec64.csv" \
  --feat_auto_pca_dim "${FEAT_PCA_DIM}" \
  "${META_CROSS_EXTRA[@]}" \
  "${TRAIN_OOF_EXTRA[@]}" \
  "${XGB_EXTRA[@]}" \
  --stacking_C 1.5 \
  --holdout-report-base-fusions \
  --also-val-pick-triple-fusion \
  --holdout-val-pick-mean3-vs-meta-val-slack "${ALSO_VAL_PICK_SLACK:-0.012}" \
  --no-append_csv \
  --out_json "$OUT_JSON" \
  --csv_row_tag_zh "${CSV_ROW_TAG_ZH_DUAL:-【域内test+独立集also·Focal+XGB+robust·RBFrefIND·pca${FEAT_PCA_DIM}·also_triple】}"

"$PY" -c "import json;d=json.load(open('$OUT_JSON'));print('official test_mcc',d['test_metrics']['mcc'],'acc',d['test_metrics']['accuracy']);ao=d.get('test_metrics_also_holdout_npz');print('independent also_mcc (meta)',ao['mcc'] if ao else None,'acc',ao['accuracy'] if ao else None);ap=d.get('test_metrics_also_extrapolation_val_picked');print('independent also_mcc (val_picked)',ap['mcc'] if ap else None,'acc',ap['accuracy'] if ap else None)"

echo "完成：$OUT_JSON（test_metrics=官方 test；test_metrics_also_holdout_npz=独立集 Meta；test_metrics_also_extrapolation_val_picked=独立集 val 三路选优）"
