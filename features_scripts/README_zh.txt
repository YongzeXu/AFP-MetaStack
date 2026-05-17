Level-1 特征缓存（公开仓需本地生成）。

主复现顺序：
  1. python features_scripts/extract_esm2_saprot_token_bins.py
     → features/afp_e_esm2_saprot_token_bins/
  2. 独立集：python features_scripts/append_independent_token_bins.py
  3. 有真实 3Di 后：python features_scripts/refresh_independent_saprot_token_bins_with_3di.py

GearNet 图嵌入缓存由 prepare_ensemble_probs_afp_e.py 训练阶段写入
features/afp_e_gearnet_mc_emb_cache/（无需单独 experimental 脚本，若使用默认 weighted_heads 分支）。
