数据目录（公开仓不含 FASTA / PDB 本体）

请按 AFPDeepPred 官方划分自行放入：
  train_set/   训练池 FASTA（及脚本需要的 3Di/PDB 等，若已具备）
  test_set/    官方测试 FASTA
  independent_set/  独立评测 FASTA，见 independent_set/INDEPENDENT_PREP.txt

特征缓存写入 features/；prepare 与 stacking 产物写入 trained/（见各目录 README）。

环境：见 env/environment_setup.txt（推荐环境名 afp-metastack，可自定）。

环境变量（可选）：
  AFP_MODELS_ROOT   HuggingFace / SaProt / GearNet 权重根目录（默认 ~/models）
  PYTHON            覆盖 pipeline shell 使用的 Python 解释器
