本仓库提供用于正文评估的 FASTA 数据划分：

- train_set：训练池，7,334 条 AFP 和 8,482 条非 AFP。
- test_set：正文 Test set 1，800 条 AFP 和 957 条非 AFP。
- independent_set：同源性筛查后的正文 Test set 2，80 条 AFP 和 73 条非 AFP。

每个划分均包含正、负类 FASTA。独立集的 id_map.json 将原始划分 ID 映射到发布版 FASTA 使用的序列 ID。

本数据集发布只包含 FASTA 和独立集 ID 映射，不包含 PDB 结构、3Di 文件或 missing_pdb_ids.txt。PDB/3Di 属于结构特征输入，可按正文方法生成；missing_pdb_ids.txt 是本地处理记录，不是评估数据或运行输入。

特征缓存写入 features/；prepare 与 stacking 产物写入 trained/。环境设置见 env/environment_setup.txt。可通过 AFP_MODELS_ROOT 指定 ESM-2、SaProt 和 GearNet 权重目录（默认 ~/models）。
