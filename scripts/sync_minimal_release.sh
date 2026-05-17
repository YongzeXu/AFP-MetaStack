#!/usr/bin/env bash
# Copy only minimal-reproduction files from AFP_E into this repo.
set -euo pipefail
SRC="${AFP_E_ROOT:-/data/yongze/AFP_E}"
DST="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d "$SRC/classifier_scripts" ]]; then
  echo "Set AFP_E_ROOT to your full AFP_E path" >&2
  exit 1
fi

copy() { mkdir -p "$(dirname "$2")" && cp -f "$1" "$2"; }

for f in binary_focal_loss.py prepare_ensemble_probs_afp_e.py ensemble_stacking_plus_abc.py \
  functional_meta_csv.py train_esm2_plus_gearnet_concat_cv10.py train_esm2_gearnet_multiview_mtl_cv10.py; do
  copy "$SRC/classifier_scripts/$f" "$DST/classifier_scripts/$f"
done
for f in run_prepare_focal_val_test_official.sh run_independent_eval.sh \
  run_focal_lift_independent_pipeline.sh run_focal_official_test_and_independent_xgb_dual_eval.sh; do
  copy "$SRC/classifier_scripts/$f" "$DST/classifier_scripts/$f"
  sed -i 's|^ROOT="/data/yongze/AFP-MetaStack"|ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." \&\& pwd)"|' \
    "$DST/classifier_scripts/$f" 2>/dev/null || true
done
for f in extract_esm2_saprot_token_bins.py append_independent_token_bins.py \
  refresh_independent_saprot_token_bins_with_3di.py; do
  copy "$SRC/features_scripts/$f" "$DST/features_scripts/$f"
done
copy "$SRC/processing/build_independent_3di_foldseek.py" "$DST/processing/build_independent_3di_foldseek.py" 2>/dev/null || true
copy "$SRC/data/README_zh.txt" "$DST/data/README_zh.txt"
copy "$SRC/data/independent_set/INDEPENDENT_PREP.txt" "$DST/data/independent_set/INDEPENDENT_PREP.txt" 2>/dev/null || true
for f in environment_setup.txt environment.yml; do
  copy "$SRC/env/$f" "$DST/env/$f" 2>/dev/null || true
done

echo "Synced minimal release → $DST ($(find "$DST" -type f | wc -l) files)"
