# AFP-MetaStack (minimal release)

Leakage-aware **two-level stacking** for antifreeze protein classification: ESM-2 + SaProt + GearNet (Level-1) → Meta-XGBoost (Level-2).

This repository includes the benchmark FASTA splits used in the manuscript and the feature-extraction and evaluation code. It does not include PDB structures, 3Di inputs, extracted feature caches, NPZ probability files, model checkpoints, paper figures, or ablation suites. Structural inputs and model features can be regenerated following the manuscript methods.

## Layout

```
classifier_scripts/     # prepare NPZ + Meta-XGB stacking + 4 entry shell scripts
features_scripts/       # ESM-2 / SaProt token-bin extraction (3 scripts)
processing/             # optional: independent-set 3Di via Foldseek (1 script)
data/                   # train/test/independent FASTA splits (see data/README_zh.txt)
features/               # generated caches (empty by default)
trained/                # val/test/independent NPZ + stacking JSON (empty by default)
env/                    # conda notes
```

## Quick start (after data are in place)

1. Create conda env — `env/environment.yml` and `env/environment_setup.txt` (`conda activate afp-metastack`)
2. The benchmark FASTA splits are included under `data/` (see `data/README_zh.txt`)
3. Set `AFP_MODELS_ROOT` (default `~/models`) for ESM-2 / SaProt / GearNet weights
4. Prepare structural inputs and build feature caches following the manuscript methods and scripts under `features_scripts/`
5. **One-shot main pipeline** (Focal γ=1.5, dual official + independent Meta-XGB):

```bash
bash classifier_scripts/run_focal_lift_independent_pipeline.sh
```

This runs, in order: official val/test prepare → independent prepare → independent Meta-XGB → dual eval JSON with both official test and independent holdout metrics.

The evaluation scripts also expect the GO/PPI feature CSV inputs documented in their command-line arguments. Those derived files are not part of this dataset release.

Equivalent manual steps:

```bash
bash classifier_scripts/run_prepare_focal_val_test_official.sh
# align independent features, then:
bash classifier_scripts/run_independent_eval.sh   # set VAL_NPZ / ENSEMBLE_INDEP_OUT_DIR / PREPARE_EXTRA_ARGS per script header
bash classifier_scripts/run_focal_official_test_and_independent_xgb_dual_eval.sh
```

Default outputs (γ=1.5):

- `trained/ensemble_prep_weighted_heads_gear_npz_focal_g1p5/{val,test}_probs_for_ensemble.npz`
- `trained/ensemble_prep_independent_weighted_heads_focal_g1p5/test_probs_for_ensemble.npz`
- `trained/ensemble_prep_weighted_heads_gear_npz_focal_g1p5/stacking_official_test_and_independent_focal_g1p5_xgb_robust_rbf_g0003_reglambda15.json`

Internal validation: 15% stratified hold-out from training pool (`random_state=2026`), same as the manuscript.

**Reference metrics** (paper run, read from stacking JSON after reproduction): official test Acc 97.21%, MCC 0.944, AUC 99.61%; independent holdout Acc 90.85%, MCC 0.820, AUC 94.01%.

## Dataset splits

- `data/train_set/`: training pool (7,334 AFPs and 8,482 non-AFPs)
- `data/test_set/`: manuscript test set 1 (800 AFPs and 957 non-AFPs)
- `data/independent_set/`: homology-screened manuscript test set 2 (80 AFPs and 73 non-AFPs)

Each split includes class-specific FASTA files. The independent set also includes `id_map.json`, which maps original split identifiers to the IDs used in the released FASTA files. PDB and 3Di files are omitted from this dataset release; regenerate them as described in the manuscript methods when needed for feature extraction.

## What is not included in the repository

- `figure/`, `ablation/`, `configs/`, `classifier_scripts/experimental/`
- Extra tuning shells (`run_two_branch_*`, train_oof, etc.)
- PDB structures, 3Di files, BLAST reference trees, and generated feature caches

## Citation & license

Cite the AFP-MetaStack manuscript. Benchmark splits follow [AFPDeepPred](https://doi.org/10.1021/acs.jcim.5c02024). License: MIT (`LICENSE`).
