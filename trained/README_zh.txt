训练与 stacking 产物目录（公开仓不含二进制文件）。

主复现默认路径（Focal γ=1.5）：
  ensemble_prep_weighted_heads_gear_npz_focal_g1p5/
    val_probs_for_ensemble.npz
    test_probs_for_ensemble.npz
    stacking_official_test_and_independent_focal_g1p5_xgb_robust_rbf_g0003_reglambda15.json
  ensemble_prep_independent_weighted_heads_focal_g1p5/
    test_probs_for_ensemble.npz

可选：mean pLDDT 表（Meta 默认参数可能引用）
  ensemble_prep_stacking_row19/mean_plddt_from_pdb.csv
  若缺失，请自行生成/准备该 CSV，或在 stacking 中关闭相关特征。
