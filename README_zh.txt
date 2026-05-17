AFP-MetaStack 最小公开仓（仅主复现链路）

约 25 个文件：prepare + stacking 核心 Python、4 条 pipeline 脚本、3 个特征脚本。

环境：conda env create -f env/environment.yml → conda activate afp-metastack（详见 env/environment_setup.txt）

数据与权重需本地准备：
  data/      ← AFPDeepPred 官方 train/test + independent FASTA
  features/  ← extract_esm2_saprot_token_bins.py 等生成
  trained/   ← prepare 与 stacking 写出

一键主流程：
  bash classifier_scripts/run_focal_lift_independent_pipeline.sh
