#!/usr/bin/env bash
# 在「特征与独立集（mapping / GO / PPI 等）已按 INDEPENDENT_PREP 对齐」后：导出独立集三路概率 NPZ，并用与域内 stacking 相同的官方 val_npz 跑 Meta（holdout）。
# 前置：append_independent_token_bins、refresh_gearnet、map+merge UniProt、refresh_independent_functional_vectors 等（见 data/independent_set/INDEPENDENT_PREP.txt）。
# 可选：WITH_HOLDOUT_3DI=1 时在 Meta 中并入独立集 Foldseek 3Di（写出 stacking_independent_with_3di.json；默认主 JSON 不带此项，见 HOLDOUT_3DI_EXPERIMENT_20260510.txt）。
# 若域内底层为 Focal：先跑 classifier_scripts/run_prepare_focal_val_test_official.sh 生成 val_npz，
# 再设 VAL_NPZ 指向该目录；并对独立集 prepare 传入相同损失，例如：
#   export VAL_NPZ="${ROOT}/trained/ensemble_prep_weighted_heads_gear_npz_focal_g15/val_probs_for_ensemble.npz"
#   export ENSEMBLE_INDEP_OUT_DIR="${ROOT}/trained/ensemble_prep_independent_weighted_heads_focal_g15"
#   export PREPARE_EXTRA_ARGS="--loss focal --focal_gamma 1.5"
#   export ROBUST_OOD_META=1   # Meta 层加 pLDDT 门控×三路p + PPI 行统计（与 Focal 概率组合时常抬独立集）
#   bash classifier_scripts/run_independent_eval.sh
# 推荐：conda activate afp-metastack（见 env/environment_setup.txt）；未 activate 时使用 CONDA_PREFIX 或 python3。
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
VAL_NPZ="${VAL_NPZ:-${ROOT}/trained/ensemble_prep_weighted_heads_gear_npz/val_probs_for_ensemble.npz}"
OUT_DIR="${ENSEMBLE_INDEP_OUT_DIR:-${ROOT}/trained/ensemble_prep_independent_weighted_heads}"
OUT_JSON="${STACKING_OUT_JSON:-}"
if [[ -z "${OUT_JSON}" ]]; then
  if [[ "${WITH_HOLDOUT_3DI:-0}" == "1" ]]; then
    OUT_JSON="${OUT_DIR}/stacking_independent_with_3di.json"
  else
    OUT_JSON="${OUT_DIR}/stacking_independent.json"
  fi
fi
mkdir -p "$OUT_DIR"
_prep_extra=()
if [[ -n "${PREPARE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  _prep_extra=( ${PREPARE_EXTRA_ARGS} )
fi

_stack_robust=()
if [[ "${ROBUST_OOD_META:-0}" == "1" ]]; then
  _stack_robust=(--robust_ood_meta)
fi

if [[ ! -f "$VAL_NPZ" ]]; then
  echo "缺少官方 val NPZ: $VAL_NPZ（请先跑 classifier_scripts/run_prepare_focal_val_test_official.sh，或从备份恢复）" >&2
  exit 1
fi

IND_3DI="${ROOT}/data/independent_set/3di/independent_3di.fasta"
echo ">>> Foldseek 3Di（独立集 PDB → 3Di FASTA；见 processing/build_independent_3di_foldseek.py）"
"$PY" processing/build_independent_3di_foldseek.py

HOLDOUT_3DI_ARGS=()
if [[ "${WITH_HOLDOUT_3DI:-0}" == "1" ]]; then
  HOLDOUT_3DI_ARGS=(--holdout_3di_fasta "$IND_3DI")
  if [[ -z "${STACKING_OUT_JSON:-}" ]]; then
    OUT_JSON="${OUT_DIR}/stacking_independent_with_3di.json"
  fi
  echo ">>> 已设 WITH_HOLDOUT_3DI=1：Meta 将用独立集 3Di 算 missing_ratio，结果写入 ${OUT_JSON}（默认主文件仍为无 holdout 3Di）"
fi

echo ">>> prepare phase=test（10 折在 train 上重训，仅写 test NPZ，耗时可较长）"
"$PY" classifier_scripts/prepare_ensemble_probs_afp_e.py \
  --gearnet_branch weighted_heads \
  --esm2_cache_dir "${ROOT}/features/afp_e_esm2_saprot_token_bins" \
  --gearnet_emb_cache_dir "${ROOT}/features/afp_e_gearnet_mc_emb_cache" \
  --out_dir "$OUT_DIR" \
  --phase test \
  --test_pos_fasta "${ROOT}/data/independent_set/pos_independent.fasta" \
  --test_neg_fasta "${ROOT}/data/independent_set/neg_independent.fasta" \
  "${_prep_extra[@]}"

HOLD="${OUT_DIR}/test_probs_for_ensemble.npz"
if [[ ! -f "$HOLD" ]]; then
  echo "未生成 $HOLD" >&2
  exit 1
fi

echo ">>> ensemble stacking（holdout = 独立集）"
# val 三路选优：Meta / mean3 / median3 在 val 上比 MCC（mean3 与 median3 先加 slack），JSON 中 test_metrics_extrapolation_val_picked 为推荐外推头
"$PY" classifier_scripts/ensemble_stacking_plus_abc.py \
  --four_panel \
  "${_stack_robust[@]}" \
  --val_npz "$VAL_NPZ" \
  --holdout_npz "$HOLD" \
  --holdout_pos_fasta "${ROOT}/data/independent_set/pos_independent.fasta" \
  --holdout_neg_fasta "${ROOT}/data/independent_set/neg_independent.fasta" \
  "${HOLDOUT_3DI_ARGS[@]}" \
  --feat_go_csv "${ROOT}/trained/functional_csv_real/feat_go_uniprot_top128.csv" \
  --feat_ppi_csv "${ROOT}/trained/functional_csv_real/feat_ppi_string_node2vec64.csv" \
  --feat_auto_pca_dim 24 \
  --stacking_C 1.5 \
  --holdout-report-base-fusions \
  --holdout-val-pick-triple-fusion \
  --holdout-val-pick-mean3-vs-meta-val-slack 0.012 \
  --holdout-zero-feat-csv-tags go,ppi \
  --no-append_csv \
  --out_json "$OUT_JSON" \
  --csv_row_tag_zh "${CSV_ROW_TAG_ZH:-【独立集·val三路选优median3mean3Meta·slack0.012·weighted_heads·gp·LR·C1.5·pca24】}"

echo "完成：${OUT_JSON} — 主 Meta 见 test_metrics；外推推荐见 test_metrics_extrapolation_val_picked"
