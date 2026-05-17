#!/usr/bin/env python3
"""
在「三路概率 Stacking」同一份 val/test NPZ 上，将 Meta-LR 输入扩展为：

  (A) k-mer：氨基酸 3-mer 频率向量经 PCA 压缩（在验证集上 fit，避免维数爆炸）；
  (B) 概率 Hadamard：p_esm*p_saprot, p_esm*p_gearnet, p_saprot*p_gearnet；
  (C) 缺结构比例：3Di 覆盖相对序列长度（补结构前的代理）；
  (图1) 分歧度：三路边际概率的标准差、两两绝对差、与「均匀共识」方向的余弦背离；
  (图1) logit 空间：两两 logit 差；logit 比值（clip）；
  (图2) 理化锚点：pI（Biopython）、Kyte–Doolittle 平均疏水性、苏氨酸占比；
  (图3) 可选 mean pLDDT：CSV 提供 seq_id→pLDDT，缺失则用 val 上中位数填充；
  (图4) Meta-LR 可选 L1 正则 / 特征标准化（缓解过拟合、稀疏系数）。

需 NPZ 含 seq_ids 与 p_esm, p_saprot, p_gearnet, y（与 prepare_ensemble 一致）。

独立集 / 外推集：可传 --holdout_npz 替代官方 test NPZ 仅作第二段打分，并配合
--holdout_pos_fasta / --holdout_neg_fasta（及可选 --holdout_3di_fasta）；k-mer PCA 与 Meta 仍在 --val_npz 上 fit。

  另：--also-holdout-npz 与主 --test_npz **同时**使用时，主表 test_metrics 仍为官方 test；
  JSON 另附 test_metrics_also_holdout_npz（同一 Meta / 同一 val 选参如 blend w，仅换 Xt），**勿与 --holdout_npz 同开**。
  可选 --also-val-pick-triple-fusion + --holdout-report-base-fusions：在 val 上按与 holdout triple 相同规则选 Meta/mean3/median3，
  将 also 上对应指标写入 test_metrics_also_extrapolation_val_picked（与独立集 run_independent_focal_xgb_robust_eval 外推口径对齐）。
  --export-split-scores-dir DIR：写出 val/test/（also 时）also 的 seq_id,y,p_meta；可与 classifier_scripts/experimental/blend_meta_split_scores_val_pick.py
    联用在 **val 标签** 上选两路 Meta 概率凸组合 β，再报告 test/also（β 常在端点，未必优于单路）。
  --export-meta-shap-bundle DIR：meta_classifier=xgboost 时写出 Meta-XGB 与缩放后特征等，供 figure/plot_meta_shap_xgb.py 离线作图（XGBoost pred_contribs）。

  A–E 功能特征（可选）：用离线工具生成「seq_id + 数值列」的 CSV，通过下列参数接入 Meta-LR：
    --feat_go_csv（GO/EC/DeepGOPlus/ProteBERT 等嵌入）
    --feat_ppi_csv（STRING+PPI 的 Node2Vec/GraphSAGE 等）
    --feat_pssm_csv（PSSM/HMM：列如整体均值、平均熵等统计量）
    --feat_deeploc_csv（DeepLoc 各类别概率）
    --feat_binding_csv（GraphBind 等结合位点密度/曲线采样）
  宽表可用 --feat_auto_pca_dim 在验证集上 PCA 压缩（0 表示不自动 PCA）。

扩展（PRISM / 冲 0.97 试验）：
  --meta_classifier {lr,xgboost}：Meta 层用 LogisticRegression 或 XGBoost（需 pip install xgboost）。
  --meta-xgb-sample-weight-missing-alpha：仅 val 上 fit XGB 时按 missing_ratio 加权（不用独立集标签）。
  --meta-xgb-bootstrap-k：val 上 bootstrap 训练多个 XGB，推理时对正类概率取平均（方差抑制）。
  --meta-xgb-seed-ensemble-k：在 **完整 val 行** 上训练 K 个 Meta-XGB（random_state 取 bootstrap_seed, seed+1, …），
    推理时对正类概率取平均；与 bootstrap 行重采样 **勿同开**。用于 subsample/colsample 随机性下的方差抑制，或略抬 holdout。
  --meta_cross_features：在已拼接 go+ppi 功能块后，追加两列
    (p_esm * mean(PPI_PCA行), p_saprot * GO_PCA第1维)，需同时指定 --feat_go_csv 与 --feat_ppi_csv；
    与 --meta-val-rbf-reference-holdout-npz 同开时参考集也会拼这两列（RBF canvas 维数与 val/test 一致）。
  --meta_moe_missing：按 missing_ratio 线性软门控双 Meta-LR（低缺失 / 高缺失两 expert，
    各在 val 上 sample_weight 强调对应 regime），p=(1-g)*p_rich + g*p_poor；仅 meta_classifier=lr。
  --robust_ood_meta：外推稳健 Meta 特征（需 --fig_extra/--four_panel）：Sigmoid 门控
    σ((pLDDT−center)/scale) 与 p_gearnet/p_saprot/p_esm 的乘积项；若提供 --feat_ppi_csv 则再拼接
    原始 PPI 嵌入行的 L2 范数、平均 |·|、近零 stub 指示（不替代 PCA 块，仅作去噪补充）。
  --afp_local_motif：在 Meta 输入末尾追加固定维「Thr/TxT 等局部序列统计」（非 CGR），需序列来自
    train/test FASTA 及 holdout FASTA 的 seq_map；与 four_panel 同开时常用于外推补局部信息。
  --meta_seq_len：在 motif 之后、robust 之前追加 4 维「序列长度门控」：log1p(L)、L/val_中位长度
    （中位数仅在 val 的 seq_id 上统计，test/holdout 共用该尺度）、log1p(L)*p_esm、log1p(L)*p_gearnet。
  --threshold-scan：在验证集正类概率上扫描阈值（0.01–0.99），目标由 --threshold-scan-objective 指定
    （mcc 或 acc），并在 JSON 的 threshold_scan_report 中附加 val / test(holdout) 在该阈值下的分类指标
    （AUC/AUPR 仍用概率；与默认 0.5 阈对照）。
    若与 --platt-calibrate 同开，则在 **Platt 校准后** 的 val 概率上扫阈值。
  --platt-calibrate：在 **val** 上对 Meta 输出做 Platt（sigmoid 仿射）：单 Meta-LR 时用其 decision_function
    作为标量特征；MoE 混合概率或 XGB 时用 logit(正类概率)。校准器仅在 val 上 fit，再作用于 val/test(holdout)；
    主表 test_* 与 JSON 中 val_metrics_in_sample / test_metrics 均基于 **校准后** 概率（JSON 内另附未校准对照）。
  --export_scores_csv PATH：将当前 **eval 侧**（官方 test 或 holdout）逐条写出 seq_id、y、p_meta、子特征；
    并在同目录写入 PATH 同名 stem 的 *_strata.json（按序列长度四分位、p_meta 分箱、missing 分层的 n/acc/MCC）。
  --meta-drop-hadamard：Meta 输入去掉 **3 维概率 Hadamard**（p_esm·p_saprot、p_esm·p_gearnet、p_saprot·p_gearnet），
    用于 val 上消融；**不在独立集上调此开关**（仅作对照实验并一次性报告）。
  --meta_mean3_blend：在 **Meta 正类概率** 与 **三路算术平均概率** 之间做凸组合；混合权重 **w 仅在 val 上** 网格搜索
    使 MCC 最大，且强制 **val_MCC(w) ≥ val_MCC(w=0) − slack**（默认 slack=0.005），平局取 **更小 w**，以减轻拖垮官方 test 的风险；
    再对 test/holdout 用同一 w。**在 --platt-calibrate 之前**应用（若同开 Platt，则校准的是混合后概率）。
  --meta-lr-aux-xgb-blend：仅 meta_classifier=lr 时，在 **同特征矩阵** 上额外 fit 一支 **Meta-XGB**（与纯 XGB 线相同的 RBF/missing 加权规则），
    再在 val 上网格搜索 **p=(1-w)*p_LR+w*p_XGB**（规则与 meta_mean3_blend 相同：val MCC 最大、floor、平局取更小 w），
    用于缓解「域内 test 上 LR 更好、独立集上 XGB 更稳」的分裂；**勿与 --meta_mean3_blend 或 --meta_moe_missing 同开**。
  --meta-lr-fuse-primary-after-aux-blend：在选出 w 之后，再在 val 上对 **p=(1-α)p_blend+α·p_LR_meta** 做第二次网格（α∈[0,a_max]），
    若 val 上 α>0 能抬 MCC 则同时改写 test/also 概率；当前默认配置下常选 α=0（与仅 aux 混合等价）。
  --holdout-zero-feat-csv-tags：仅 holdout 时，将 Meta 输入中 go/ppi 等功能 CSV 对齐块置零再预测，JSON 增加
    test_metrics_meta_holdout_feat_csv_zeroed_uncalibrated（主 test_metrics 不变），用于诊断「独立集上映射差拖垮 Meta」。
  --holdout-report-base-fusions：仅 holdout 时，JSON 增加 test_metrics_prob_mean3 / median3（无 Meta），与主指标对照。
  --holdout-val-pick-mean3-vs-meta：仅 holdout：在 val 上按 MCC 比较 Meta 与 mean3，写入 test_metrics_extrapolation_val_picked（须同开 base-fusions）。
  --holdout-val-pick-triple-fusion：仅 holdout：在 val 上比较 Meta / mean3 / median3（mean3 与 median3 的得分 = MCC + val_slack），
    平局时优先 median3、再 mean3、再 Meta；写入 test_metrics_extrapolation_val_picked。勿与 --holdout-val-pick-mean3-vs-meta 同开。
  --holdout-val-pick-mean3-vs-meta-val-slack：与上两项之一联用；triple 时仅加在 mean3 与 median3 的 val 得分上。
  Meta-LR（单模型或 MoE 的 rich/poor）训练后会在 JSON 中写入 meta_lr_coef_report：各特征块的 |w| 聚合，
    用于判断局部 motif 等块是否被实际使用。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb  # type: ignore[import-not-found]
except ImportError:
    xgb = None  # type: ignore[misc, assignment]

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError:
    ProteinAnalysis = None  # type: ignore[misc, assignment]

import train_esm2_plus_saprot_token_crossattn_cv10 as tc
from functional_meta_csv import align_csv_to_splits, align_csv_val_test_oof

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
KMER_K = 3
KMER_DIM = len(AA) ** KMER_K

# Kyte–Doolittle hydropathy (approximate GRAVY when averaged over sequence)
KD_SCALE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def phys_three(seq: str) -> Tuple[float, float, float]:
    """pI, mean Kyte–Doolittle, Thr fraction."""
    u = "".join(c for c in seq.upper() if c in AA_INDEX)
    if len(u) == 0:
        return 7.0, 0.0, 0.0
    gravy = float(np.mean([KD_SCALE.get(a, 0.0) for a in u]))
    t_frac = float(u.count("T")) / float(len(u))
    if ProteinAnalysis is not None:
        try:
            pi = float(ProteinAnalysis(u).isoelectric_point())
        except Exception:
            pi = 7.0
    else:
        pi = 7.0
    return pi, gravy, t_frac


AFP_LOCAL_MOTIF_DIM = 12


def local_afp_motif_feats(seq: str) -> np.ndarray:
    """非 CGR：Thr/TxT 相关局部统计，固定 12 维，供 Meta-LR 使用。"""
    u = "".join(c for c in seq.upper() if c in AA_INDEX)
    L = len(u)
    if L == 0:
        return np.zeros(AFP_LOCAL_MOTIF_DIM, dtype=np.float32)
    t_frac = float(u.count("T")) / float(L)
    s_frac = float(u.count("S")) / float(L)
    tt = sum(1 for i in range(L - 1) if u[i : i + 2] == "TT")
    tt_n = float(tt) / float(max(L - 1, 1))
    ttt = sum(1 for i in range(L - 2) if u[i : i + 3] == "TTT")
    ttt_n = float(ttt) / float(max(L - 2, 1))
    tst_like = sum(1 for i in range(L - 2) if u[i : i + 3] in ("TST", "TSN", "TSS", "TTS"))
    tst_n = float(tst_like) / float(max(L - 2, 1))
    max_run = cur = 0
    for c in u:
        if c == "T":
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    max_run_n = float(max_run) / float(L)
    txt = sum(1 for i in range(L - 2) if u[i] == "T" and u[i + 2] == "T")
    txt_n = float(txt) / float(max(L - 2, 1))
    head = u[: min(30, L)]
    t_head = float(head.count("T")) / float(len(head))
    polar = set("STNQ")
    t_polar = sum(
        1
        for i, ch in enumerate(u)
        if ch == "T" and ((i > 0 and u[i - 1] in polar) or (i < L - 1 and u[i + 1] in polar))
    )
    t_polar_n = float(t_polar) / float(L)
    ivl = float(sum(ch in "IVL" for ch in u)) / float(L)
    de = float(sum(ch in "DE" for ch in u)) / float(L)
    kr = float(sum(ch in "KR" for ch in u)) / float(L)
    return np.array(
        [t_frac, s_frac, tt_n, ttt_n, tst_n, max_run_n, txt_n, t_head, t_polar_n, ivl, de, kr],
        dtype=np.float32,
    )


META_SEQ_LEN_DIM = 4


def meta_seq_len_gate_row(seq: str, p_esm: float, p_gearnet: float, val_median_aa_len: float) -> np.ndarray:
    """长度归一化 + 与 ESM/Gear 概率的交互；val_median_aa_len 仅在验证集上估计，避免用 test 统计。"""
    u = "".join(c for c in seq.upper() if c in AA_INDEX)
    L = float(len(u))
    med = max(float(val_median_aa_len), 1.0)
    log_l = float(np.log1p(L))
    ratio = L / med
    return np.array(
        [log_l, ratio, log_l * float(p_esm), log_l * float(p_gearnet)],
        dtype=np.float32,
    )


def prob_disagreement_and_logit(pe: np.ndarray, ps: np.ndarray, pg: np.ndarray) -> np.ndarray:
    """分歧度 + logit 差/比值（与三路概率对齐）。"""
    pe = pe.astype(np.float64)
    ps = ps.astype(np.float64)
    pg = pg.astype(np.float64)
    P = np.stack([pe, ps, pg], axis=1)
    p_std = P.std(axis=1)
    p_rng = P.max(axis=1) - P.min(axis=1)
    d_es = np.abs(pe - ps)
    d_eg = np.abs(pe - pg)
    d_sg = np.abs(ps - pg)
    # 与均匀共识 [1,1,1] 方向的「背离」（归一化余弦距离）
    v = np.stack([pe, ps, pg], axis=1).astype(np.float64)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    u = v / n
    c = np.ones((v.shape[0], 3), dtype=np.float64) / (3.0 ** 0.5)
    cos = (u * c).sum(axis=1).clip(-1.0, 1.0)
    cons_dis = 1.0 - cos

    Le, Ls, Lg = safe_logit(pe), safe_logit(ps), safe_logit(pg)
    ld_es = Le - Ls
    ld_eg = Le - Lg
    ld_sg = Ls - Lg
    eps = 1e-3
    lr_es = Le / (np.abs(Ls) + eps)
    lr_eg = Le / (np.abs(Lg) + eps)
    lr_sg = Ls / (np.abs(Lg) + eps)
    lr_es = np.clip(lr_es, -30.0, 30.0)
    lr_eg = np.clip(lr_eg, -30.0, 30.0)
    lr_sg = np.clip(lr_sg, -30.0, 30.0)

    return np.column_stack(
        [
            p_std,
            p_rng,
            d_es,
            d_eg,
            d_sg,
            cons_dis,
            ld_es,
            ld_eg,
            ld_sg,
            lr_es,
            lr_eg,
            lr_sg,
        ]
    ).astype(np.float32)


def load_plddt_map(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.is_file():
        return {}
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        low = first.lower()
        if "seq_id" in low or "sid" in low or "mean_plddt" in low or "plddt" in low:
            reader = csv.DictReader(f)
            for r in reader:
                sid = str(r.get("seq_id") or r.get("id") or r.get("SID") or "").strip()
                if not sid:
                    continue
                for k in ("mean_plddt", "avg_plddt", "plddt", "pLDDT", "mean_pLDDT"):
                    if k in r and r[k] not in (None, ""):
                        try:
                            out[sid] = float(r[k])
                        except ValueError:
                            pass
                        break
        else:
            for row in csv.reader(f):
                if len(row) >= 2:
                    try:
                        out[str(row[0]).strip()] = float(row[1])
                    except ValueError:
                        continue
    return out


def kmer_freq_vec(seq: str) -> np.ndarray:
    v = np.zeros(KMER_DIM, dtype=np.float64)
    u = "".join(c for c in seq.upper() if c in AA_INDEX)
    for i in range(len(u) - KMER_K + 1):
        mer = u[i : i + KMER_K]
        idx = 0
        ok = True
        for c in mer:
            if c not in AA_INDEX:
                ok = False
                break
            idx = idx * len(AA) + AA_INDEX[c]
        if ok:
            v[idx] += 1.0
    s = float(v.sum())
    if s > 0:
        v /= s
    return v.astype(np.float32)


def append_experiments_table_row(experiments_csv: Path, zh_row: Dict[str, str]) -> None:
    fieldnames = [
        "特征", "分类器", "融合器", "结构分支",
        "test_accuracy", "test_mcc", "test_auc", "test_aupr",
        "test_sensitivity", "test_recall", "test_specificity", "test_f1",
    ]
    rows: List[Dict[str, str]] = []
    if experiments_csv.is_file() and experiments_csv.stat().st_size > 0:
        with experiments_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            for r in reader:
                rows.append({k: (r.get(k) or "") for k in fieldnames})
    for fn in fieldnames:
        if fn not in zh_row:
            zh_row[fn] = ""
    rows.append(zh_row)
    with experiments_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_bundle(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    need = ["p_esm", "p_gearnet", "y"]
    for k in need:
        if k not in data.files:
            raise KeyError(f"{path} 缺少 {k}")
    out: Dict[str, np.ndarray] = {k: np.asarray(data[k]).reshape(-1) for k in need}
    if "p_saprot" in data.files:
        out["p_saprot"] = np.asarray(data["p_saprot"]).reshape(-1)
    if "seq_ids" not in data.files:
        raise KeyError(f"{path} 需要 seq_ids 以对齐序列计算 k-mer / 缺失比例")
    out["seq_ids"] = np.asarray(data["seq_ids"])
    n = out["y"].shape[0]
    if out["seq_ids"].shape[0] != n:
        raise ValueError("seq_ids 与 y 长度不一致")
    return out


def bundle_has_p_saprot(bundle: Dict[str, np.ndarray]) -> bool:
    return "p_saprot" in bundle


def prob_disagreement_and_logit_two_branch(pe: np.ndarray, pg: np.ndarray) -> np.ndarray:
    """两路 p_esm、p_gearnet 时的分歧/logit 特征（8 维）。"""
    pe = pe.astype(np.float64)
    pg = pg.astype(np.float64)
    P = np.stack([pe, pg], axis=1)
    p_std = P.std(axis=1)
    p_rng = np.abs(pe - pg)
    v = np.stack([pe, pg], axis=1).astype(np.float64)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    u = v / n
    c = np.ones((v.shape[0], 2), dtype=np.float64) / (2.0**0.5)
    cos = (u * c).sum(axis=1).clip(-1.0, 1.0)
    cons_dis = 1.0 - cos
    Le, Lg = safe_logit(pe), safe_logit(pg)
    ld = Le - Lg
    eps = 1e-3
    lr = np.clip(Le / (np.abs(Lg) + eps), -30.0, 30.0)
    prod = pe * pg
    return np.column_stack([p_std, p_rng, np.abs(pe - pg), cons_dis, Le, Lg, ld, prod]).astype(np.float32)


def _plddt_values(sids: List[str], plddt_map: Dict[str, float], fill: float) -> np.ndarray:
    return np.array([plddt_map.get(sid, fill) for sid in sids], dtype=np.float32)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def ppi_raw_row_stats_matrix(sids: List[str], csv_path: Path, eps: float = 1e-5) -> np.ndarray:
    """每行：(L2 范数, mean|·|, 近零 stub 指示)，用于 STRING 孤立点等外推去噪补充。"""
    n = len(sids)
    out = np.zeros((n, 3), dtype=np.float32)
    if not csv_path.is_file():
        out[:, 2] = 1.0
        return out
    sid_to_vec: Dict[str, np.ndarray] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames
        if not fns:
            out[:, 2] = 1.0
            return out
        sid_key = fns[0]
        num_keys = [k for k in fns[1:] if k]
        for row in reader:
            sid = str(row.get(sid_key, "")).strip()
            if not sid:
                continue
            vals: List[float] = []
            for k in num_keys:
                cell = row.get(k, "")
                if cell is None or str(cell).strip() == "":
                    continue
                try:
                    vals.append(float(cell))
                except ValueError:
                    continue
            if vals:
                sid_to_vec[sid] = np.asarray(vals, dtype=np.float64)
    for i, sid in enumerate(sids):
        v = sid_to_vec.get(str(sid))
        if v is None or v.size == 0:
            out[i, 2] = 1.0
            continue
        l2 = float(np.linalg.norm(v))
        ma = float(np.mean(np.abs(v)))
        out[i, 0] = np.float32(l2)
        out[i, 1] = np.float32(ma)
        out[i, 2] = np.float32(1.0 if l2 < eps else 0.0)
    return out


def build_X(
    bundle: Dict[str, np.ndarray],
    seq_map: Dict[str, str],
    all_3di: Dict[str, str],
    pca: PCA | None,
    *,
    fit_pca: bool,
    fig_extra: bool,
    plddt_map: Dict[str, float],
    plddt_fill: float,
    include_prob_hadamard_pairs: bool = True,
) -> np.ndarray:
    pe = bundle["p_esm"].astype(np.float64)
    pg = bundle["p_gearnet"].astype(np.float64)
    two = not bundle_has_p_saprot(bundle)
    if two:
        if include_prob_hadamard_pairs:
            inter = (pe * pg).astype(np.float32).reshape(-1, 1)
        else:
            inter = np.zeros((len(pe), 0), dtype=np.float32)
        parts0: List[np.ndarray] = [pe.reshape(-1, 1), pg.reshape(-1, 1)]
    else:
        ps = bundle["p_saprot"].astype(np.float64)
        inter = np.column_stack([pe * ps, pe * pg, ps * pg]).astype(np.float32)
        parts0 = [pe.reshape(-1, 1), ps.reshape(-1, 1), pg.reshape(-1, 1)]

    sids = [str(s) for s in bundle["seq_ids"].tolist()]
    km = np.stack([kmer_freq_vec(seq_map[sid]) for sid in sids], axis=0)
    if fit_pca:
        assert pca is not None
        km_p = pca.fit_transform(km)
    else:
        assert pca is not None
        km_p = pca.transform(km)

    miss = np.zeros((len(sids),), dtype=np.float32)
    for i, sid in enumerate(sids):
        seq = seq_map.get(sid, "")
        seq_len = max(1, len(seq))
        obs = min(len(all_3di.get(sid, "")), seq_len)
        miss[i] = float(max(seq_len - obs, 0)) / float(seq_len)
    phy = np.zeros((len(sids), 3), dtype=np.float32)
    if fig_extra:
        for i, sid in enumerate(sids):
            pi, gr, tf = phys_three(seq_map.get(sid, ""))
            phy[i, 0] = float(pi)
            phy[i, 1] = float(gr)
            phy[i, 2] = float(tf)

    parts: List[np.ndarray] = parts0
    if include_prob_hadamard_pairs:
        parts.append(inter)
    parts.extend(
        [
            km_p.astype(np.float32),
            miss.reshape(-1, 1),
        ]
    )
    if fig_extra:
        if two:
            parts.append(prob_disagreement_and_logit_two_branch(pe, pg))
        else:
            ps = bundle["p_saprot"].astype(np.float64)
            parts.append(prob_disagreement_and_logit(pe, ps, pg))
        parts.append(phy)
        pld = _plddt_values(sids, plddt_map, plddt_fill).reshape(-1, 1)
        parts.append(pld)

    X = np.concatenate(parts, axis=1)
    return X.astype(np.float32)


def metrics_block(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float | None = None) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    thr = 0.5 if threshold is None else float(threshold)
    y_pred = (y_prob >= thr).astype(np.int64)
    acc = float(metrics.accuracy_score(y_true, y_pred))
    mcc = float(metrics.matthews_corrcoef(y_true, y_pred))
    try:
        auc = float(metrics.roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    pc, rc, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = float(metrics.auc(rc, pc))
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(metrics.f1_score(y_true, y_pred))
    return {
        "accuracy": acc,
        "mcc": mcc,
        "auc": auc,
        "aupr": aupr,
        "sensitivity": sens,
        "recall": sens,
        "specificity": spec,
        "f1": f1,
    }


def meta_lr_column_blocks(
    *,
    kmer_pca_dim: int,
    fig_extra: bool,
    func_metas: List[Dict[str, object]],
    meta_cross: bool,
    motif_dim: int,
    seq_len_dim: int,
    robust_dim: int,
    include_hadamard: bool = True,
    n_base_probs: int = 3,
    hadamard_width: int = 3,
    fig_disagreement_cols: int = 12,
) -> List[Tuple[str, int, int]]:
    """与 main 中 X 拼接顺序一致，用于解释 Meta-LR 系数。"""
    i = 0
    blocks: List[Tuple[str, int, int]] = []
    tag = "three_base_probs" if n_base_probs == 3 else "two_base_probs"
    blocks.append((tag, i, i + n_base_probs))
    i += n_base_probs
    if include_hadamard:
        blocks.append(("prob_hadamard_pairs", i, i + hadamard_width))
        i += hadamard_width
    blocks.append((f"kmer_pca_{kmer_pca_dim}", i, i + kmer_pca_dim))
    i += kmer_pca_dim
    blocks.append(("missing_ratio", i, i + 1))
    i += 1
    if fig_extra:
        blocks.append((f"disagreement_logit_{fig_disagreement_cols}", i, i + fig_disagreement_cols))
        i += fig_disagreement_cols
        blocks.append(("phys_pi_gravy_thr_frac", i, i + 3))
        i += 3
        blocks.append(("mean_plddt", i, i + 1))
        i += 1
    for m in func_metas:
        tag = str(m.get("tag", "func"))
        d = int(m.get("dim_out", 0))
        if d > 0:
            pca_tag = "_pca" if m.get("pca") else ""
            blocks.append((f"feat_csv_{tag}{pca_tag}_{d}", i, i + d))
            i += d
    if meta_cross:
        blocks.append(("meta_cross_2col", i, i + 2))
        i += 2
    if motif_dim > 0:
        blocks.append((f"afp_local_motif_{motif_dim}", i, i + motif_dim))
        i += motif_dim
    if seq_len_dim > 0:
        blocks.append((f"meta_seq_len_gate_{seq_len_dim}", i, i + seq_len_dim))
        i += seq_len_dim
    if robust_dim > 0:
        blocks.append((f"robust_ood_meta_{robust_dim}", i, i + robust_dim))
        i += robust_dim
    return blocks


def summarize_lr_coef_blocks(coef: np.ndarray, blocks: List[Tuple[str, int, int]]) -> List[Dict[str, object]]:
    w = np.asarray(coef, dtype=np.float64).reshape(-1)
    rows: List[Dict[str, object]] = []
    for name, lo, hi in blocks:
        seg = w[lo:hi]
        rows.append(
            {
                "block": name,
                "slice": [lo, hi],
                "dim": hi - lo,
                "l1": float(np.sum(np.abs(seg))),
                "l2": float(np.linalg.norm(seg)),
                "linf": float(np.max(np.abs(seg))) if seg.size > 0 else 0.0,
            }
        )
    return rows


def scan_threshold_max_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """在 val 上粗扫阈值，使 MCC 最大；返回 (best_threshold, best_mcc)。"""
    y_true = y_true.astype(np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    best_thr, best_mcc = 0.5, -2.0
    for thr in np.linspace(0.01, 0.99, 199):
        m = float(metrics.matthews_corrcoef(y_true, (p >= thr).astype(np.int64)))
        if m > best_mcc:
            best_mcc, best_thr = m, float(thr)
    return best_thr, best_mcc


def scan_threshold_max_acc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """在 val 上粗扫阈值，使 accuracy 最大；返回 (best_threshold, best_acc)。"""
    y_true = y_true.astype(np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    best_thr, best_acc = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 199):
        pred = (p >= thr).astype(np.int64)
        acc = float(np.mean(pred == y_true))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, best_acc


def platt_fit_apply(
    *,
    prob_va_uncal: np.ndarray,
    prob_te_uncal: np.ndarray,
    yv: np.ndarray,
    yt: np.ndarray,
    clf: object,
    Xv_fit: np.ndarray,
    Xt_fit: np.ndarray,
    meta_cls: str,
    meta_moe_missing: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """在 val 上拟合 Platt（1 维 logistic on score），再变换 val/test 正类概率。"""
    pu = np.asarray(prob_va_uncal, dtype=np.float64).reshape(-1)
    pt = np.asarray(prob_te_uncal, dtype=np.float64).reshape(-1)
    use_df = meta_cls == "lr" and not meta_moe_missing and hasattr(clf, "decision_function")
    if use_df:
        z_v = np.asarray(clf.decision_function(Xv_fit), dtype=np.float64).reshape(-1, 1)
        z_t = np.asarray(clf.decision_function(Xt_fit), dtype=np.float64).reshape(-1, 1)
        feat_name = "meta_lr_decision_function"
    else:
        z_v = safe_logit(pu).reshape(-1, 1)
        z_t = safe_logit(pt).reshape(-1, 1)
        feat_name = "logit_meta_positive_prob"
    cal = LogisticRegression(C=1e12, max_iter=5000, random_state=2026, solver="lbfgs")
    cal.fit(z_v, yv.astype(np.int64))
    pva = cal.predict_proba(z_v)[:, 1].astype(np.float64)
    pte = cal.predict_proba(z_t)[:, 1].astype(np.float64)
    aux: Dict[str, object] = {
        "platt_feature": feat_name,
        "platt_coef_a": float(np.asarray(cal.coef_).reshape(-1)[0]),
        "platt_intercept_b": float(np.asarray(cal.intercept_).reshape(-1)[0]),
        "val_uncalibrated_metrics_0p5": metrics_block(yv, pu),
        "eval_uncalibrated_metrics_0p5": metrics_block(yt.astype(np.int64), pt),
    }
    return pva, pte, aux


def _safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(metrics.matthews_corrcoef(y_true, y_pred))


def _write_sid_y_p_csv(path: Path, sids: List[str], y: np.ndarray, p: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_flat = np.asarray(y, dtype=np.int64).reshape(-1)
    p_flat = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(sids) != int(y_flat.size) or len(sids) != int(p_flat.size):
        raise SystemExit(f"内部错误：_write_sid_y_p_csv 长度不一致 n_sid={len(sids)} ny={y_flat.size} np={p_flat.size}")
    with path.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["seq_id", "y", "p_meta"])
        for i, sid in enumerate(sids):
            wr.writerow([sid, int(y_flat[i]), float(p_flat[i])])


def _stratum_stats(yt: np.ndarray, p_meta: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    yt = np.asarray(yt, dtype=np.int64).reshape(-1)
    p_meta = np.asarray(p_meta, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    y_s, p_s = yt[m], p_meta[m]
    pred = (p_s >= 0.5).astype(np.int64)
    mcc_v = _safe_mcc(y_s, pred)
    mcc_out: Optional[float] = float(mcc_v) if np.isfinite(mcc_v) else None
    return {
        "n": n,
        "n_positive": int(y_s.sum()),
        "accuracy": float((pred == y_s).mean()),
        "mcc": mcc_out,
        "mean_p_meta": float(np.mean(p_s)),
    }


def export_eval_score_profile(
    csv_path: Path,
    sids: List[str],
    yt: np.ndarray,
    prob_te: np.ndarray,
    bundle: Dict[str, np.ndarray],
    Xt_unscaled: np.ndarray,
    miss_col: int,
    seq_map: Dict[str, str],
) -> Dict[str, object]:
    """逐条写 CSV，并返回分层统计。"""
    yt = np.asarray(yt, dtype=np.int64).reshape(-1)
    p = np.asarray(prob_te, dtype=np.float64).reshape(-1)
    pe = bundle["p_esm"].astype(np.float64).reshape(-1)
    ps = bundle["p_saprot"].astype(np.float64).reshape(-1) if bundle_has_p_saprot(bundle) else np.full(pe.shape, np.nan, dtype=np.float64)
    pg = bundle["p_gearnet"].astype(np.float64).reshape(-1)
    miss = Xt_unscaled[:, miss_col].astype(np.float64).reshape(-1)
    lens = np.array(
        [len("".join(c for c in seq_map.get(sid, "").upper() if c in AA_INDEX)) for sid in sids],
        dtype=np.int32,
    )
    pred = (p >= 0.5).astype(np.int64)
    correct = (pred == yt).astype(np.int32)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seq_id",
                "y_true",
                "p_meta",
                "y_pred",
                "correct",
                "p_esm",
                "p_saprot",
                "p_gearnet",
                "seq_len_aa",
                "missing_ratio",
            ]
        )
        for i, sid in enumerate(sids):
            w.writerow(
                [
                    sid,
                    int(yt[i]),
                    f"{p[i]:.8f}",
                    int(pred[i]),
                    int(correct[i]),
                    f"{pe[i]:.8f}",
                    ("" if np.isnan(ps[i]) else f"{ps[i]:.8f}"),
                    f"{pg[i]:.8f}",
                    int(lens[i]),
                    f"{miss[i]:.8f}",
                ]
            )

    out: Dict[str, object] = {
        "overall": _stratum_stats(yt, p, np.ones_like(yt, dtype=bool)),
    }

    q = np.quantile(lens.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
    len_rows: List[Dict[str, object]] = []
    for b in range(4):
        lo, hi = float(q[b]), float(q[b + 1])
        if b == 3:
            mask = (lens.astype(np.float64) >= lo) & (lens.astype(np.float64) <= hi + 1e-9)
        else:
            mask = (lens.astype(np.float64) >= lo) & (lens.astype(np.float64) < hi)
        row = _stratum_stats(yt, p, mask)
        row["seq_len_bin"] = b + 1
        row["seq_len_lo_inclusive"] = lo
        row["seq_len_hi"] = hi
        len_rows.append(row)
    out["by_seq_len_quartile_on_eval"] = len_rows

    pq = np.quantile(p, np.linspace(0.0, 1.0, 11))
    p_rows: List[Dict[str, object]] = []
    for b in range(10):
        lo, hi = float(pq[b]), float(pq[b + 1])
        if b == 9:
            mask = (p >= lo) & (p <= hi + 1e-15)
        else:
            mask = (p >= lo) & (p < hi)
        row = _stratum_stats(yt, p, mask)
        row["p_meta_decile"] = b + 1
        row["p_lo"] = lo
        row["p_hi"] = hi
        p_rows.append(row)
    out["by_p_meta_decile_on_eval"] = p_rows

    m_edges = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 + 1e-9]
    m_labels = ["missing_low", "missing_mid", "missing_high"]
    m_rows: List[Dict[str, object]] = []
    for b in range(3):
        lo, hi = m_edges[b], m_edges[b + 1]
        mask = (miss >= lo) & (miss < hi)
        row = _stratum_stats(yt, p, mask)
        row["missing_tertile"] = m_labels[b]
        m_rows.append(row)
    out["by_missing_ratio_tertile"] = m_rows

    return out


def apply_meta_mean3_blend_tune(
    *,
    prob_va: np.ndarray,
    prob_te: np.ndarray,
    yv: np.ndarray,
    val_bundle: Dict[str, np.ndarray],
    test_bundle: Dict[str, np.ndarray],
    w_max: float,
    n_grid: int,
    mcc_slack: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """p=(1-w)*p_meta+w*mean(三路p)；w 仅在 val 上网格搜索最大 MCC，且 val_MCC>=MCC(w=0)-slack；平局取更小 w。"""
    pv = np.asarray(prob_va, dtype=np.float64).reshape(-1)
    pt = np.asarray(prob_te, dtype=np.float64).reshape(-1)
    pe_v = val_bundle["p_esm"].astype(np.float64).reshape(-1)
    pg_v = val_bundle["p_gearnet"].astype(np.float64).reshape(-1)
    pe_t = test_bundle["p_esm"].astype(np.float64).reshape(-1)
    pg_t = test_bundle["p_gearnet"].astype(np.float64).reshape(-1)
    if bundle_has_p_saprot(val_bundle):
        ps_v = val_bundle["p_saprot"].astype(np.float64).reshape(-1)
        ps_t = test_bundle["p_saprot"].astype(np.float64).reshape(-1)
        mv = (pe_v + ps_v + pg_v) / 3.0
        mt = (pe_t + ps_t + pg_t) / 3.0
    else:
        mv = (pe_v + pg_v) / 2.0
        mt = (pe_t + pg_t) / 2.0
    yv_i = np.asarray(yv, dtype=np.int64).reshape(-1)
    base_mcc = float(metrics.matthews_corrcoef(yv_i, (pv >= 0.5).astype(np.int64)))
    floor_mcc = base_mcc - float(mcc_slack)
    n_g = max(2, int(n_grid))
    grid = np.linspace(0.0, float(w_max), n_g)
    best_w, best_mcc = 0.0, base_mcc
    for w in grid:
        wf = float(w)
        pb = (1.0 - wf) * pv + wf * mv
        mcc = float(metrics.matthews_corrcoef(yv_i, (pb >= 0.5).astype(np.int64)))
        if mcc < floor_mcc - 1e-15:
            continue
        if mcc > best_mcc + 1e-12:
            best_mcc, best_w = mcc, wf
        elif abs(mcc - best_mcc) <= 1e-12 and wf < best_w:
            best_w = wf
    pb_va = (1.0 - best_w) * pv + best_w * mv
    pb_te = (1.0 - best_w) * pt + best_w * mt
    info: Dict[str, object] = {
        "meta_mean3_blend_w": best_w,
        "val_mcc_meta_only": base_mcc,
        "val_mcc_after_blend": best_mcc,
        "constraint_val_mcc_floor": floor_mcc,
        "w_max": float(w_max),
        "n_grid": n_g,
    }
    return pb_va, pb_te, info


def apply_meta_aux_prob_blend_tune(
    *,
    prob_va_primary: np.ndarray,
    prob_te_primary: np.ndarray,
    prob_va_aux: np.ndarray,
    prob_te_aux: np.ndarray,
    yv: np.ndarray,
    w_max: float,
    n_grid: int,
    mcc_slack: float,
    val_select_penalty_lambda: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """p=(1-w)*p_primary+w*p_aux；w 在 val 上网格搜索。

    默认：最大化 val MCC（且 val_MCC≥MCC(w=0)-slack；MCC 平局取更小 w）。
    若 val_select_penalty_lambda>0：最大化 (val_MCC - λ·w)，在同一 floor 约束下，倾向更小 w，便于在「官方 test 偏 LR」
    与「独立集偏 XGB」之间折中（仍不使用 holdout 标签）。
    """
    pv = np.asarray(prob_va_primary, dtype=np.float64).reshape(-1)
    pt = np.asarray(prob_te_primary, dtype=np.float64).reshape(-1)
    av = np.asarray(prob_va_aux, dtype=np.float64).reshape(-1)
    at = np.asarray(prob_te_aux, dtype=np.float64).reshape(-1)
    yv_i = np.asarray(yv, dtype=np.int64).reshape(-1)
    base_mcc = float(metrics.matthews_corrcoef(yv_i, (pv >= 0.5).astype(np.int64)))
    floor_mcc = base_mcc - float(mcc_slack)
    n_g = max(2, int(n_grid))
    grid = np.linspace(0.0, float(w_max), n_g)
    pen = float(val_select_penalty_lambda or 0.0)
    best_w, best_mcc = 0.0, base_mcc
    best_score = base_mcc - pen * 0.0
    for w in grid:
        wf = float(w)
        pb = (1.0 - wf) * pv + wf * av
        mcc = float(metrics.matthews_corrcoef(yv_i, (pb >= 0.5).astype(np.int64)))
        if mcc < floor_mcc - 1e-15:
            continue
        score = mcc - pen * wf
        if score > best_score + 1e-12:
            best_score, best_mcc, best_w = score, mcc, wf
        elif abs(score - best_score) <= 1e-12 and wf < best_w:
            best_w, best_mcc = wf, mcc
    pb_va = (1.0 - best_w) * pv + best_w * av
    pb_te = (1.0 - best_w) * pt + best_w * at
    info: Dict[str, object] = {
        "blend_mode": "lr_primary_xgb_aux",
        "meta_lr_aux_xgb_blend_w": best_w,
        "val_mcc_primary_only": base_mcc,
        "val_mcc_after_aux_xgb_blend": best_mcc,
        "constraint_val_mcc_floor": floor_mcc,
        "val_select_penalty_lambda": pen,
        "val_blend_select_score": best_score,
        "w_max": float(w_max),
        "n_grid": n_g,
    }
    return pb_va, pb_te, info


def apply_meta_fuse_primary_after_aux_blend_tune(
    *,
    prob_va_blend: np.ndarray,
    prob_te_blend: np.ndarray,
    prob_va_lr_meta: np.ndarray,
    prob_te_lr_meta: np.ndarray,
    yv: np.ndarray,
    a_max: float,
    n_grid: int,
    mcc_slack: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """在 aux 混合概率与「同一 Meta-LR」概率间再做凸组合：p=(1-α)p_blend+α·p_LR_meta，α∈[0,a_max] 仅在 val 上选。

    动机：p_blend 常更利 also；p_LR_meta 常更利官方 test；二次融合可能在两边同时略抬（不保证）。
    """
    pb_v = np.asarray(prob_va_blend, dtype=np.float64).reshape(-1)
    pb_t = np.asarray(prob_te_blend, dtype=np.float64).reshape(-1)
    pl_v = np.asarray(prob_va_lr_meta, dtype=np.float64).reshape(-1)
    pl_t = np.asarray(prob_te_lr_meta, dtype=np.float64).reshape(-1)
    yv_i = np.asarray(yv, dtype=np.int64).reshape(-1)
    base_mcc = float(metrics.matthews_corrcoef(yv_i, (pb_v >= 0.5).astype(np.int64)))
    floor_mcc = base_mcc - float(mcc_slack)
    n_g = max(2, int(n_grid))
    amax = float(np.clip(float(a_max), 0.0, 1.0))
    grid = np.linspace(0.0, amax, n_g)
    best_a, best_mcc = 0.0, base_mcc
    for a in grid:
        af = float(a)
        pv = (1.0 - af) * pb_v + af * pl_v
        mcc = float(metrics.matthews_corrcoef(yv_i, (pv >= 0.5).astype(np.int64)))
        if mcc < floor_mcc - 1e-15:
            continue
        if mcc > best_mcc + 1e-12:
            best_mcc, best_a = mcc, af
        elif abs(mcc - best_mcc) <= 1e-12 and af < best_a:
            best_a = af
    out_v = (1.0 - best_a) * pb_v + best_a * pl_v
    out_t = (1.0 - best_a) * pb_t + best_a * pl_t
    info: Dict[str, object] = {
        "fuse_mode": "blend_then_lr_meta_convex",
        "meta_lr_fuse_primary_after_aux_blend_alpha": best_a,
        "val_mcc_at_alpha0_aux_blend_only": base_mcc,
        "val_mcc_after_fuse": best_mcc,
        "constraint_val_mcc_floor_vs_alpha0": floor_mcc,
        "alpha_max": float(amax),
        "n_grid": n_g,
    }
    return out_v, out_t, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="三路概率 Meta-LR + k-mer PCA + 概率交互 + 缺失比例")
    p.add_argument(
        "--val_npz",
        type=Path,
        default=ROOT / "trained" / "ensemble_prep_stacking_row19" / "val_probs_for_ensemble.npz",
    )
    p.add_argument(
        "--test_npz",
        type=Path,
        default=ROOT / "trained" / "ensemble_prep_stacking_row19" / "test_probs_for_ensemble.npz",
    )
    p.add_argument(
        "--train_oof_npz",
        type=Path,
        default=None,
        help="可选：prepare_ensemble_probs_afp_e.py --phase train_oof 写出的 train_oof_probs_for_ensemble.npz。"
        "与 val_npz 同源划分时 seq_id 会重叠；脚本会自动去掉与 val 重复的 id，仅把剩余 OOF 行并入 Meta 拟合（StandardScaler 仍在 val 上 fit；"
        "val 上 in-sample 概率与 JSON 仍以 val_npz 行为准）。可与 --meta_cross_features 同开（OOF 行使用对齐后的 GO/PPI 块算交叉列）。",
    )
    p.add_argument("--kmer_pca_dim", type=int, default=32, help="k-mer(8000) PCA 到该维数")
    p.add_argument("--stacking_C", type=float, default=1.0)
    p.add_argument(
        "--stacking_penalty",
        type=str,
        default="l2",
        choices=("l1", "l2"),
        help="Meta-LR 正则：图4 建议可试 L1 稀疏系数",
    )
    p.add_argument(
        "--scale_meta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="对 Meta 输入做 StandardScaler（在 val 上 fit），常与 L1 同用以平衡量纲",
    )
    p.add_argument(
        "--fig_extra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用图1–3：分歧度+logit 比值/差、理化锚点、可选 pLDDT",
    )
    p.add_argument(
        "--plddt_csv",
        type=Path,
        default=None,
        help="可选：seq_id→mean pLDDT（表头含 seq_id 与 mean_plddt 等）；无则该维用 val 上中位数常数",
    )
    p.add_argument("--out_json", type=Path, default=None)
    p.add_argument("--experiments_csv", type=Path, default=ROOT / "results" / "experiments_table_zh.csv")
    p.add_argument(
        "--append_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否往 experiments_table_zh.csv 追加一行",
    )
    p.add_argument(
        "--four_panel",
        action="store_true",
        help="四图要点一键：--fig_extra + --stacking_penalty l1；若存在 mean_plddt_from_pdb.csv 则自动作 --plddt_csv",
    )
    p.add_argument(
        "--feat_go_csv",
        type=Path,
        default=None,
        help="A: GO/EC 或功能文本嵌入（seq_id + 数值列）",
    )
    p.add_argument(
        "--feat_ppi_csv",
        type=Path,
        default=None,
        help="B: STRING/PPI 图嵌入（seq_id + 数值列）",
    )
    p.add_argument(
        "--feat_pssm_csv",
        type=Path,
        default=None,
        help="C: PSSM/HMM 统计量（如列均值、列熵等，seq_id + 数值列）",
    )
    p.add_argument(
        "--feat_deeploc_csv",
        type=Path,
        default=None,
        help="D: DeepLoc 等亚细胞定位概率向量（seq_id + 各类概率）",
    )
    p.add_argument(
        "--feat_binding_csv",
        type=Path,
        default=None,
        help="E: 配体/辅因子结合位点密度等（seq_id + 数值列）",
    )
    p.add_argument(
        "--feat_auto_pca_dim",
        type=int,
        default=32,
        help="功能块列数大于该值时，在 val 上 PCA 压缩到该维数；0=从不自动 PCA",
    )
    p.add_argument(
        "--csv_row_tag_zh",
        type=str,
        default="",
        help="非空时：在写入 experiments_table_zh 的「特征」列最前面追加该标记（便于消融/对照）",
    )
    p.add_argument(
        "--meta_classifier",
        type=str,
        choices=("lr", "xgboost"),
        default="lr",
        help="Meta 分类器：lr=LogisticRegression；xgboost=梯度提升树（非线性）",
    )
    p.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=220,
        help="meta_classifier=xgboost 时 XGBClassifier.n_estimators",
    )
    p.add_argument(
        "--xgb-max-depth",
        type=int,
        default=3,
        help="meta_classifier=xgboost 时 XGBClassifier.max_depth",
    )
    p.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.06,
        help="meta_classifier=xgboost 时 XGBClassifier.learning_rate",
    )
    p.add_argument(
        "--xgb-reg-lambda",
        type=float,
        default=2.5,
        help="meta_classifier=xgboost 时 XGBClassifier.reg_lambda",
    )
    p.add_argument(
        "--xgb-min-child-weight",
        type=float,
        default=4.0,
        help="meta_classifier=xgboost 时 XGBClassifier.min_child_weight",
    )
    p.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.85,
        help="meta_classifier=xgboost 时 XGBClassifier.subsample",
    )
    p.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.85,
        help="meta_classifier=xgboost 时 XGBClassifier.colsample_bytree",
    )
    p.add_argument(
        "--xgb-scale-pos-weight",
        type=float,
        default=None,
        help="若给定：传入 XGBClassifier(scale_pos_weight=…)；>1 更重视正类。与 --xgb-auto-scale-pos-weight 勿同开。",
    )
    p.add_argument(
        "--xgb-auto-scale-pos-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="为 True 时：在 val 上按 n_neg/n_pos 自动设 scale_pos_weight（仅 Meta-XGB fit）。",
    )
    p.add_argument(
        "--meta-xgb-sample-weight-missing-alpha",
        type=float,
        default=0.0,
        help=">0：仅在 val 上 fit Meta-XGB 时使用 sample_weight∝1+alpha*(missing/median(missing))；"
        "强调高 missing 子集，意在域外稳健；未使用独立集标签。",
    )
    p.add_argument(
        "--meta-train-oof-augment-row-weight",
        type=float,
        default=1.0,
        help="与 --train_oof_npz 同开且 Meta-XGB（或 Meta-LR）在拼接矩阵上 fit 时：对 **OOF 尾部行** 的 sample_weight 乘该因子（0<w≤1），"
        "减轻低质量/少 epoch OOF 三路概率对 Meta 的拉扯；默认 1.0 不改变行为。",
    )
    p.add_argument(
        "--meta-xgb-bootstrap-k",
        type=int,
        default=1,
        help="在 val 上训练 K 个 Meta-XGB：K=1 为单模型；K>1 时为 bootstrap 行重采样后各训一模型，"
        "val/test/holdout 正类概率为 K 模型平均。",
    )
    p.add_argument(
        "--meta-xgb-bootstrap-seed",
        type=int,
        default=2026,
        help="bootstrap 的 numpy RandomState 种子；亦作 --meta-xgb-seed-ensemble-k>1 时首棵 XGB 的 random_state",
    )
    p.add_argument(
        "--meta-xgb-seed-ensemble-k",
        type=int,
        default=1,
        help="在完整 val 上训练 K 个 Meta-XGB（random_state=bootstrap_seed..bootstrap_seed+K-1），概率取平均；"
        "与 --meta-xgb-bootstrap-k>1 勿同开",
    )
    p.add_argument(
        "--meta-val-reweight-holdout-rbf-gamma",
        type=float,
        default=0.0,
        help=">0：在 val 上 fit Meta 前赋 sample_weight，权重与「val 行到 canvas 的 RBF 平均核」成正比；"
        "canvas 为 --holdout_npz 的 Meta 特征，或（无 holdout 时）--meta-val-rbf-reference-holdout-npz 的 Meta 特征；"
        "不读 canvas 集标签。Meta-XGB 的 fit 使用该权重；Meta-LR 默认不乘该权重（见 --meta-lr-use-rbf-sample-weight），"
        "MoE-LR 亦同。γ 越小核越宽、权重更平滑，可网格扫描（勿在最终 test 上调参）。",
    )
    p.add_argument(
        "--meta-val-reweight-holdout-alpha",
        type=float,
        default=0.5,
        help="与 --meta-val-reweight-holdout-rbf-gamma 联用：w ∝ 1 + alpha * sim/mean(sim)",
    )
    p.add_argument(
        "--meta-val-reweight-holdout-max-rows",
        type=int,
        default=512,
        help="holdout 侧参与 RBF 的最大行数（随机子采样，控内存/时间）",
    )
    p.add_argument(
        "--meta-val-reweight-holdout-rbf-avg-seeds",
        type=int,
        default=1,
        help=">1：用多个 RandomState(meta_xgb_bootstrap_seed + k) 各算一遍 RBF 权重后取算术平均再归一化，"
        "降低 max-rows 子采样方差；仅影响 val 上 Meta-XGB 的 sample_weight，不读标签。",
    )
    p.add_argument(
        "--meta-val-rbf-reference-holdout-npz",
        type=Path,
        default=None,
        help="与 --holdout_npz 二选一：仅用于 RBF 权重的「无标签」参考集 NPZ（如独立集）；"
        "主评估仍为 --test_npz（如官方 test）。勿与 --holdout_npz 同开；"
        "与 reference 同开时请关闭 --afp_local_motif、--meta_seq_len（参考集未拼该块）；"
        "--meta_cross_features 可与 reference 同开（参考集会拼相同交叉列）。",
    )
    p.add_argument(
        "--meta-val-rbf-reference-pos-fasta",
        type=Path,
        default=None,
        help="与 --meta-val-rbf-reference-holdout-npz 联用：参考集正例 FASTA",
    )
    p.add_argument(
        "--meta-val-rbf-reference-neg-fasta",
        type=Path,
        default=None,
        help="与 --meta-val-rbf-reference-holdout-npz 联用：参考集负例 FASTA",
    )
    p.add_argument(
        "--meta-val-rbf-reference-3di-fasta",
        type=Path,
        default=None,
        help="可选：参考集 3Di FASTA",
    )
    p.add_argument(
        "--meta_cross_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="追加 p_esm*mean(PPI_PCA)、p_saprot*GO_PCA1 两列；需 go+ppi 两 CSV",
    )
    p.add_argument(
        "--meta_moe_missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="missing_ratio 软门控双 Meta-LR（低缺失/高缺失加权两 expert）；仅 meta_classifier=lr",
    )
    p.add_argument("--moe_gate_low", type=float, default=0.1, help="gate→0 端：missing 低于此更信 rich expert")
    p.add_argument("--moe_gate_high", type=float, default=0.5, help="gate→1 端：missing 高于此更信 poor expert")
    p.add_argument(
        "--moe_expert_boost",
        type=float,
        default=3.0,
        help="各 expert 在 val 上 fit 时的 sample_weight：对应 regime 内为该倍数，其余为 1",
    )
    p.add_argument(
        "--holdout_npz",
        type=Path,
        default=None,
        help="若指定：用该 NPZ 替代「官方 test_npz」仅作第二段打分（如独立测试集）；"
        "val 与 k-mer PCA 仍在 val_npz 上 fit；Meta 仍在 val 上 fit。需同时提供 holdout FASTA 以补全序列。",
    )
    p.add_argument(
        "--holdout_pos_fasta",
        type=Path,
        default=None,
        help="与 --holdout_npz 联用：holdout 正例 FASTA（seq_id 与 NPZ 中 seq_ids 一致）",
    )
    p.add_argument(
        "--holdout_neg_fasta",
        type=Path,
        default=None,
        help="与 --holdout_npz 联用：holdout 负例 FASTA",
    )
    p.add_argument(
        "--holdout_3di_fasta",
        type=Path,
        default=None,
        help="可选：holdout 的 3Di FASTA；不提供则 missing_ratio 按无 3Di 处理",
    )
    p.add_argument(
        "--also-holdout-npz",
        type=Path,
        default=None,
        help="与主 --test_npz 同跑：官方 test 仍写 test_metrics；另对 also NPZ 算 test_metrics_also_holdout_npz。"
        "需 also FASTA；勿与 --holdout_npz 同开；勿与 --platt-calibrate / --meta_moe_missing / --meta_mean3_blend 同开。",
    )
    p.add_argument(
        "--also-holdout-pos-fasta",
        type=Path,
        default=None,
        help="与 --also-holdout-npz 联用：also 正例 FASTA",
    )
    p.add_argument(
        "--also-holdout-neg-fasta",
        type=Path,
        default=None,
        help="与 --also-holdout-npz 联用：also 负例 FASTA",
    )
    p.add_argument(
        "--also-holdout-3di-fasta",
        type=Path,
        default=None,
        help="可选：also 集 3Di FASTA",
    )
    p.add_argument(
        "--holdout-zero-feat-csv-tags",
        type=str,
        default="",
        help="仅 holdout：逗号分隔如 go,ppi — 在 Meta 推理前把对应 feat_csv_* PCA 列整块置零，"
        "减轻独立集 UniProt/功能映射不可靠时的误导；主 test_metrics 不变，另写 *_go_ppi_zeroed 对照。",
    )
    p.add_argument(
        "--holdout-report-base-fusions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅 holdout：在 JSON 中增加 test_metrics_prob_mean3 / test_metrics_prob_median3（无 Meta，仅三路概率融合）",
    )
    p.add_argument(
        "--holdout-val-pick-mean3-vs-meta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅 holdout：在 val 上比较 Meta 与三路 mean3 的 MCC（阈 0.5），"
        "在独立集上推荐与 val 同策略的一侧；写入 extrapolation_val_pick_* 与 test_metrics_extrapolation_val_picked。"
        "需同时开启 --holdout-report-base-fusions。",
    )
    p.add_argument(
        "--holdout-val-pick-triple-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅 holdout：在 val 上比较 Meta / mean3 / median3 的 MCC（阈 0.5）；mean3 与 median3 的 val 得分加上 "
        "--holdout-val-pick-mean3-vs-meta-val-slack 后再与 meta 比；平局优先 median3>mean3>Meta。"
        "写入 extrapolation_val_pick_report 与 test_metrics_extrapolation_val_picked。须同开 --holdout-report-base-fusions；"
        "勿与 --holdout-val-pick-mean3-vs-meta 同开。",
    )
    p.add_argument(
        "--holdout-val-pick-mean3-vs-meta-val-slack",
        type=float,
        default=0.0,
        help="与 --holdout-val-pick-mean3-vs-meta：选 mean3 当 (mean3_MCC+slack)>meta_MCC。"
        "与 --holdout-val-pick-triple-fusion：slack 加在 val 上 mean3 与 median3 的 MCC 得分上（Meta 不加）。",
    )
    p.add_argument(
        "--also-val-pick-triple-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="与 --test_npz + --also-holdout-npz 联用（勿与 --holdout_npz 同开）：在 val 上按与 "
        "--holdout-val-pick-triple-fusion 相同规则比较 Meta / mean3 / median3，将所选一侧在 also NPZ 上的指标写入 "
        "also_extrapolation_val_pick_report 与 test_metrics_also_extrapolation_val_picked；"
        "须同开 --holdout-report-base-fusions（用于在 also 上预计算 also_metrics_prob_mean3 / median3）。",
    )
    p.add_argument(
        "--robust_ood_meta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="外推稳健：pLDDT Sigmoid 门控 × 三路概率 +（若提供 feat_ppi_csv）PPI 原始行 L2/均值|·|/stub",
    )
    p.add_argument(
        "--robust_plddt_center",
        type=float,
        default=60.0,
        help="门控 σ((pLDDT−center)/scale)；pLDDT 高于 center 更信结构路概率",
    )
    p.add_argument(
        "--robust_plddt_scale",
        type=float,
        default=8.0,
        help="门控 scale；越小门控越陡",
    )
    p.add_argument(
        "--afp_local_motif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Meta 末尾追加 12 维 Thr/TxT 等局部序列统计（非 CGR）；需 seq_map 含对应 FASTA",
    )
    p.add_argument(
        "--meta_seq_len",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Meta 追加 4 维：log1p(L)、L/val 中位长度、log1p(L)*p_esm、log1p(L)*p_gearnet（在 motif 后、robust 前）",
    )
    p.add_argument(
        "--threshold-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="val 上扫阈值（0.01–0.99），目标由 --threshold-scan-objective 指定；JSON 附加 test/holdout 在该阈值下的指标（对照默认 0.5）",
    )
    p.add_argument(
        "--threshold-scan-objective",
        type=str,
        choices=("mcc", "acc"),
        default="mcc",
        help="与 --threshold-scan 联用：在 val 上选阈值的优化目标（mcc=Matthews 相关；acc=准确率）",
    )
    p.add_argument(
        "--platt-calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="val 上 Platt（sigmoid 仿射）校准 Meta 正类概率；主指标与 CSV 用校准后概率",
    )
    p.add_argument(
        "--export_scores_csv",
        type=Path,
        default=None,
        help="将 eval 侧逐条 y、p_meta、子特征写入该 CSV；同目录写 stem_strata.json 分层摘要",
    )
    p.add_argument(
        "--export-split-scores-dir",
        type=Path,
        default=None,
        help="若非空：写入 val.csv、test.csv（及 also_holdout 时 also.csv），列 seq_id,y,p_meta；"
        "供离线在 val 上搜索两路 Meta 概率凸组合（勿用 also 标签选参）。",
    )
    p.add_argument(
        "--export-meta-shap-bundle",
        type=Path,
        default=None,
        help="若非空目录：meta_classifier=xgboost 时写出 Meta-XGB 与 val/test 上缩放后特征矩阵等，"
        "供 figure/plot_meta_shap_xgb.py 离线作图（pred_contribs）。",
    )
    p.add_argument(
        "--meta-drop-hadamard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Meta 输入中去掉三路边际概率 Hadamard 交互（p_esm*p_saprot 等 3 维）；missing 列索引随之左移",
    )
    p.add_argument(
        "--meta-lr-use-rbf-sample-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="--meta-val-reweight-holdout-rbf-gamma>0 且 meta_classifier=lr 时：是否在 val 上 fit Meta-LR（含 MoE）时乘与 XGB 相同的 RBF sample_weight。"
        "域内 test 上默认关闭更稳（加权略损）；与 XGB 对齐的加权实验可加 --meta-lr-use-rbf-sample-weight。",
    )
    p.add_argument(
        "--meta_mean3_blend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="p=(1-w)*p_meta+w*mean(三路p)；w 仅在 val 上网格最大化 MCC，且 val_MCC≥MCC(w=0)-slack",
    )
    p.add_argument(
        "--meta_mean3_blend_w_max",
        type=float,
        default=0.15,
        help="w 搜索上界（默认 0.15 偏保守，减轻拖垮官方 test）；可显式加大再试",
    )
    p.add_argument(
        "--meta_mean3_blend_grid",
        type=int,
        default=16,
        help="[0,w_max] 上等分点数（≥2）",
    )
    p.add_argument(
        "--meta_mean3_blend_mcc_slack",
        type=float,
        default=0.002,
        help="相对 w=0 的 val MCC 允许的最大跌幅（默认更紧）",
    )
    p.add_argument(
        "--meta-lr-aux-xgb-blend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="meta_classifier=lr 时：额外 fit Meta-XGB，再在 val 上网格搜索 p=(1-w)p_LR+w*p_XGB（勿与 --meta_mean3_blend / --meta_moe_missing 同开）",
    )
    p.add_argument(
        "--meta-lr-aux-xgb-blend-w-max",
        type=float,
        default=1.0,
        help="上述 w 的搜索上界（1.0 允许 val 上选到纯 XGB 分支）",
    )
    p.add_argument(
        "--meta-lr-aux-xgb-blend-grid",
        type=int,
        default=51,
        help="[0,w_max] 上等分点数（≥2）。在 --meta-lr-aux-xgb-blend-val-penalty-lambda>0 时不宜盲目极大化：更细网格可能把 w 推向更偏 XGB，"
        "val 上 (MCC−λw) 略好但官方 test MCC 反降（独立 also 未必升）；默认约 0.02 步长已较稳。",
    )
    p.add_argument(
        "--meta-lr-aux-xgb-blend-mcc-slack",
        type=float,
        default=0.003,
        help="相对 w=0（仅 LR）的 val MCC 允许的最大跌幅",
    )
    p.add_argument(
        "--meta-lr-aux-xgb-blend-val-penalty-lambda",
        type=float,
        default=0.0,
        help=">0 时：val 上选 w 改为最大化 (val_MCC - λ·w)，在 floor 约束下偏好更小 w，便于官方 test 多留 LR、"
        "独立集仍保留部分 XGB；λ 需小步试（如 0.03–0.12）",
    )
    p.add_argument(
        "--meta-lr-fuse-primary-after-aux-blend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="在 --meta-lr-aux-xgb-blend 选出 w 之后：val 上再网格搜索 p=(1-α)p_blend+α·p_LR_meta（p_LR_meta 为同一 Meta-LR 正类概率），"
        "α∈[0,a_max] 最大化 val MCC；可能对官方 test 与 also 同时略抬。勿与 --platt-calibrate 同开。",
    )
    p.add_argument(
        "--meta-lr-fuse-primary-after-aux-blend-a-max",
        type=float,
        default=0.5,
        help="上述 α 的上界（含 0；建议 0.2–0.5 小步试）",
    )
    p.add_argument(
        "--meta-lr-fuse-primary-after-aux-blend-grid",
        type=int,
        default=41,
        help="α 网格点数（≥2）",
    )
    p.add_argument(
        "--meta-lr-fuse-primary-after-aux-blend-mcc-slack",
        type=float,
        default=0.0,
        help="相对 α=0（仅 aux 混合）的 val MCC 允许跌幅；默认 0",
    )
    return p.parse_args()


def _make_xgb_meta_clf(
    args: argparse.Namespace, *, scale_pos_weight: float | None = None, random_state: int = 2026
) -> object:
    if xgb is None:
        raise RuntimeError("未安装 xgboost，请 pip install xgboost")
    kw: Dict[str, object] = dict(
        n_estimators=int(args.xgb_n_estimators),
        max_depth=int(args.xgb_max_depth),
        learning_rate=float(args.xgb_learning_rate),
        min_child_weight=float(args.xgb_min_child_weight),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample_bytree),
        reg_lambda=float(args.xgb_reg_lambda),
        reg_alpha=0.15,
        objective="binary:logistic",
        random_state=int(random_state),
        n_jobs=-1,
        tree_method="hist",
    )
    if scale_pos_weight is not None:
        kw["scale_pos_weight"] = float(scale_pos_weight)
    return xgb.XGBClassifier(**kw)


def _missing_ratio_sample_weights(miss_v: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    """Meta-XGB 在 val 上 fit 用的 sample_weight；alpha<=0 则关闭。"""
    a = float(alpha)
    if a <= 0.0:
        return None
    mv = np.asarray(miss_v, dtype=np.float64).reshape(-1)
    med = float(np.median(mv)) + 1e-9
    w = 1.0 + a * (mv / med)
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float64)


def _meta_xgb_proba_positive_mean(clfs: List[object], X: np.ndarray) -> np.ndarray:
    if not clfs:
        raise ValueError("Meta-XGB 模型列表为空")
    xf = np.asarray(X, dtype=np.float32)
    parts = [c.predict_proba(xf)[:, 1].astype(np.float64) for c in clfs]
    return np.mean(np.stack(parts, axis=0), axis=0)


def _holdout_rbf_val_sample_weights(
    Xv_fit: np.ndarray,
    Xt_fit: np.ndarray,
    *,
    gamma: float,
    alpha: float,
    max_ho: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """val 行权重：与 holdout（独立集）Meta 特征分布的 RBF 相似度；不用 holdout 标签。"""
    xv = np.asarray(Xv_fit, dtype=np.float64)
    xt_full = np.asarray(Xt_fit, dtype=np.float64)
    nho = int(xt_full.shape[0])
    if max_ho > 0 and nho > max_ho:
        pick = rng.choice(nho, size=max_ho, replace=False)
        xt = xt_full[pick]
    else:
        xt = xt_full
    d2 = euclidean_distances(xv, xt, squared=True)
    sim = np.exp(-float(gamma) * d2).mean(axis=1)
    w = 1.0 + float(alpha) * (sim / (float(np.mean(sim)) + 1e-12))
    w = w / (float(np.mean(w)) + 1e-12)
    return w.astype(np.float64)


def _resolve_xgb_scale_pos_weight(args: argparse.Namespace, yv: np.ndarray) -> float | None:
    """Meta-XGB 的 scale_pos_weight；None 表示不传（库默认）。"""
    auto = bool(getattr(args, "xgb_auto_scale_pos_weight", False))
    manual = getattr(args, "xgb_scale_pos_weight", None)
    if auto and manual is not None:
        raise SystemExit("不要同时指定 --xgb-auto-scale-pos-weight 与 --xgb-scale-pos-weight")
    if auto:
        yv_i = np.asarray(yv, dtype=np.int64).reshape(-1)
        n_pos = int(np.sum(yv_i == 1))
        n_neg = int(np.sum(yv_i == 0))
        if n_pos < 1:
            return 1.0
        return float(n_neg) / float(max(n_pos, 1))
    if manual is not None:
        return float(manual)
    return None


def _fit_meta_xgb_models(
    args: argparse.Namespace,
    *,
    Xv_fit: np.ndarray,
    yv: np.ndarray,
    Xt_fit: np.ndarray,
    miss_v: np.ndarray,
    extra_val_weights: Optional[np.ndarray] = None,
    X_augment_fit: Optional[np.ndarray] = None,
    y_augment: Optional[np.ndarray] = None,
    miss_augment: Optional[np.ndarray] = None,
) -> Tuple[List[object], np.ndarray, np.ndarray]:
    """返回 (clfs, prob_va, prob_te)。bootstrap_k<=1 且 seed_ensemble_k<=1 时等价单模型。

    若提供 X_augment_fit / y_augment：在 **拼接后的矩阵** 上 fit（仍只在原始 val 行 Xv_fit 上算 prob_va）。
    """
    n0 = int(Xv_fit.shape[0])
    Xtr = np.asarray(Xv_fit, dtype=np.float32)
    ytr = np.asarray(yv, dtype=np.int64).reshape(-1)
    miss_tr = np.asarray(miss_v, dtype=np.float64).reshape(-1)
    if X_augment_fit is not None and y_augment is not None:
        Xtr = np.ascontiguousarray(np.vstack([Xv_fit, X_augment_fit.astype(np.float32)]))
        ytr = np.concatenate([ytr, np.asarray(y_augment, dtype=np.int64).reshape(-1)], axis=0)
        if miss_augment is None:
            raise SystemExit("内部错误：Meta-XGB augment 需提供 miss_augment")
        miss_tr = np.concatenate([miss_tr, np.asarray(miss_augment, dtype=np.float64).reshape(-1)], axis=0)
    sw_all = _missing_ratio_sample_weights(miss_tr, float(getattr(args, "meta_xgb_sample_weight_missing_alpha", 0.0) or 0.0))
    n_tr_total = int(Xtr.shape[0])
    if extra_val_weights is not None:
        ev = np.asarray(extra_val_weights, dtype=np.float64).reshape(-1)
        if ev.shape[0] != n0:
            raise SystemExit(f"extra_val_weights 长度 {ev.shape[0]} 与 val 行数 {n0} 不一致")
        if X_augment_fit is not None:
            tail = np.ones(int(X_augment_fit.shape[0]), dtype=np.float64)
            ev = np.concatenate([ev, tail], axis=0)
            ev = ev / (float(np.mean(ev)) + 1e-12)
        if sw_all is None:
            sw_all = np.ones(n_tr_total, dtype=np.float64)
        else:
            sw_all = np.asarray(sw_all, dtype=np.float64).reshape(-1).copy()
        sw_all = sw_all * ev
        sw_all = sw_all / (float(np.mean(sw_all)) + 1e-12)
    oof_n = int(X_augment_fit.shape[0]) if (X_augment_fit is not None and y_augment is not None) else 0
    oof_rw = float(getattr(args, "meta_train_oof_augment_row_weight", 1.0) or 1.0)
    if oof_n > 0 and oof_rw > 0.0 and abs(oof_rw - 1.0) > 1e-15:
        if sw_all is None:
            sw_all = np.ones(n_tr_total, dtype=np.float64)
        else:
            sw_all = np.asarray(sw_all, dtype=np.float64).reshape(-1).copy()
        sw_all[n0:] *= oof_rw
        sw_all = sw_all / (float(np.mean(sw_all)) + 1e-12)
    k_boot = int(getattr(args, "meta_xgb_bootstrap_k", 1) or 1)
    if k_boot < 1:
        raise SystemExit("--meta-xgb-bootstrap-k 必须 >=1")
    k_seed_ens = int(getattr(args, "meta_xgb_seed_ensemble_k", 1) or 1)
    if k_seed_ens < 1:
        raise SystemExit("--meta-xgb-seed-ensemble-k 必须 >=1")
    seed = int(getattr(args, "meta_xgb_bootstrap_seed", 2026) or 2026)
    rng = np.random.RandomState(seed)
    clfs: List[object] = []
    spw = _resolve_xgb_scale_pos_weight(args, ytr)
    if k_seed_ens > 1:
        fit_kw_full: Dict[str, object] = {}
        if sw_all is not None:
            fit_kw_full["sample_weight"] = sw_all
        for j in range(k_seed_ens):
            clf_j = _make_xgb_meta_clf(args, scale_pos_weight=spw, random_state=seed + j)
            clf_j.fit(Xtr, ytr, **fit_kw_full)
            clfs.append(clf_j)
    elif k_boot <= 1:
        clf = _make_xgb_meta_clf(args, scale_pos_weight=spw, random_state=seed)
        fit_kw: Dict[str, object] = {}
        if sw_all is not None:
            fit_kw["sample_weight"] = sw_all
        clf.fit(Xtr, ytr, **fit_kw)
        clfs = [clf]
    else:
        for _ in range(k_boot):
            idx = rng.randint(0, n_tr_total, size=n_tr_total)
            clf_i = _make_xgb_meta_clf(args, scale_pos_weight=spw, random_state=seed)
            fit_kw = {}
            if sw_all is not None:
                fit_kw["sample_weight"] = sw_all[idx]
            clf_i.fit(Xtr[idx], ytr[idx], **fit_kw)
            clfs.append(clf_i)
    prob_va = _meta_xgb_proba_positive_mean(clfs, Xv_fit)
    prob_te = _meta_xgb_proba_positive_mean(clfs, Xt_fit)
    return clfs, prob_va, prob_te


def _make_meta_clf(penalty: str, C: float) -> LogisticRegression:
    pen = str(penalty).lower()
    if pen == "l1":
        return LogisticRegression(
            max_iter=10000,
            C=float(C),
            penalty="l1",
            solver="saga",
            random_state=2026,
            n_jobs=-1,
        )
    return LogisticRegression(max_iter=5000, C=float(C), penalty="l2", solver="lbfgs")


def main() -> None:
    args = parse_args()
    default_plddt = ROOT / "trained" / "ensemble_prep_stacking_row19" / "mean_plddt_from_pdb.csv"
    if args.four_panel:
        args.fig_extra = True
        args.stacking_penalty = "l1"
        if args.plddt_csv is None and default_plddt.is_file():
            args.plddt_csv = default_plddt

    if getattr(args, "holdout_val_pick_mean3_vs_meta", False) and getattr(args, "holdout_val_pick_triple_fusion", False):
        raise SystemExit("不要同时开启 --holdout-val-pick-mean3-vs-meta 与 --holdout-val-pick-triple-fusion")
    if getattr(args, "holdout_val_pick_triple_fusion", False):
        if args.holdout_npz is None:
            raise SystemExit("--holdout-val-pick-triple-fusion 仅在与 --holdout_npz 联用时有效")
        if not bool(getattr(args, "holdout_report_base_fusions", False)):
            raise SystemExit("--holdout-val-pick-triple-fusion 需同时开启 --holdout-report-base-fusions")
    if getattr(args, "holdout_val_pick_mean3_vs_meta", False):
        if args.holdout_npz is None:
            raise SystemExit("--holdout-val-pick-mean3-vs-meta 仅在与 --holdout_npz 联用时有效")
        if not bool(getattr(args, "holdout_report_base_fusions", False)):
            raise SystemExit("--holdout-val-pick-mean3-vs-meta 需同时开启 --holdout-report-base-fusions")

    aux_lr_xgb_blend = bool(getattr(args, "meta_lr_aux_xgb_blend", False))
    if aux_lr_xgb_blend:
        if str(args.meta_classifier or "lr").lower() != "lr":
            raise SystemExit("--meta-lr-aux-xgb-blend 仅支持 meta_classifier=lr（勿与 --meta_classifier xgboost 同开）")
        if args.meta_moe_missing:
            raise SystemExit("--meta-lr-aux-xgb-blend 与 --meta_moe_missing 勿同开")
        if getattr(args, "meta_mean3_blend", False):
            raise SystemExit("--meta-lr-aux-xgb-blend 与 --meta_mean3_blend 勿同开")

    fuse_lr_after_aux = bool(getattr(args, "meta_lr_fuse_primary_after_aux_blend", False))
    if fuse_lr_after_aux:
        if not aux_lr_xgb_blend:
            raise SystemExit("--meta-lr-fuse-primary-after-aux-blend 需同时开启 --meta-lr-aux-xgb-blend")
        if bool(getattr(args, "platt_calibrate", False)):
            raise SystemExit("--meta-lr-fuse-primary-after-aux-blend 与 --platt-calibrate 勿同开")

    also_npz_p = getattr(args, "also_holdout_npz", None)
    if also_npz_p is not None:
        if args.holdout_npz is not None:
            raise SystemExit("不要同时指定 --holdout_npz 与 --also-holdout-npz")
        if getattr(args, "also_holdout_pos_fasta", None) is None or getattr(args, "also_holdout_neg_fasta", None) is None:
            raise SystemExit("--also-holdout-npz 需同时指定 --also-holdout-pos-fasta 与 --also-holdout-neg-fasta")
        if bool(getattr(args, "platt_calibrate", False)):
            raise SystemExit("--also-holdout-npz 与 --platt-calibrate 暂勿同开")
        if bool(args.meta_moe_missing):
            raise SystemExit("--also-holdout-npz 与 --meta_moe_missing 暂勿同开")
        if bool(getattr(args, "meta_mean3_blend", False)):
            raise SystemExit("--also-holdout-npz 与 --meta_mean3_blend 暂勿同开")

    if bool(getattr(args, "also_val_pick_triple_fusion", False)):
        if args.holdout_npz is not None:
            raise SystemExit("--also-val-pick-triple-fusion 与 --holdout_npz 勿同开")
        if also_npz_p is None or args.test_npz is None:
            raise SystemExit("--also-val-pick-triple-fusion 需同时指定 --test_npz 与 --also-holdout-npz")
        if not bool(getattr(args, "holdout_report_base_fusions", False)):
            raise SystemExit("--also-val-pick-triple-fusion 需同时开启 --holdout-report-base-fusions")

    if bool(getattr(args, "xgb_auto_scale_pos_weight", False)) and getattr(args, "xgb_scale_pos_weight", None) is not None:
        raise SystemExit("不要同时指定 --xgb-auto-scale-pos-weight 与 --xgb-scale-pos-weight")
    if int(getattr(args, "meta_xgb_seed_ensemble_k", 1) or 1) > 1 and int(getattr(args, "meta_xgb_bootstrap_k", 1) or 1) > 1:
        raise SystemExit("--meta-xgb-seed-ensemble-k>1 与 --meta-xgb-bootstrap-k>1 勿同开（二者均为多模型平均，语义冲突）")

    train_oof_npz_p = getattr(args, "train_oof_npz", None)
    if train_oof_npz_p is not None:
        rw = float(getattr(args, "meta_train_oof_augment_row_weight", 1.0) or 1.0)
        if rw <= 0.0 or rw > 1.0 + 1e-9:
            raise SystemExit("--meta-train-oof-augment-row-weight 必须在 (0, 1]（默认 1.0 表示不降权）")
        if bool(args.afp_local_motif):
            raise SystemExit("--train-oof-npz 与 --afp_local_motif 暂不兼容")
        if bool(args.meta_seq_len):
            raise SystemExit("--train-oof-npz 与 --meta_seq_len 暂不兼容")
        if bool(args.meta_moe_missing):
            raise SystemExit("--train-oof-npz 与 --meta_moe_missing 暂不兼容")

    ref_npz_p = getattr(args, "meta_val_rbf_reference_holdout_npz", None)
    if ref_npz_p is not None:
        if args.holdout_npz is not None:
            raise SystemExit("不要同时指定 --holdout_npz 与 --meta-val-rbf-reference-holdout-npz")
        if bool(args.afp_local_motif) or bool(args.meta_seq_len):
            raise SystemExit(
                "与 --meta-val-rbf-reference-holdout-npz 联用时请关闭 --afp_local_motif、--meta_seq_len"
                "（参考集侧尚未拼接对应块）；--meta_cross_features 已支持并与参考集对齐。"
            )
        if getattr(args, "meta_val_rbf_reference_pos_fasta", None) is None or getattr(
            args, "meta_val_rbf_reference_neg_fasta", None
        ) is None:
            raise SystemExit("reference 需同时指定 --meta-val-rbf-reference-pos-fasta 与 --meta-val-rbf-reference-neg-fasta")

    val = load_bundle(args.val_npz.resolve())
    train_oof_bundle: Optional[Dict[str, np.ndarray]] = None
    y_oof: Optional[np.ndarray] = None
    train_oof_raw_n = 0
    train_oof_val_overlap_dropped = 0
    if train_oof_npz_p is not None:
        raw = load_bundle(train_oof_npz_p.resolve())
        val_sid_set = {str(s) for s in val["seq_ids"].tolist()}
        s_all = [str(s) for s in raw["seq_ids"].tolist()]
        train_oof_raw_n = len(s_all)
        mask = np.array([sid not in val_sid_set for sid in s_all], dtype=bool)
        train_oof_val_overlap_dropped = int(np.sum(~mask))
        if int(mask.sum()) < 1:
            raise SystemExit(
                "--train-oof-npz 与 --val-npz 去重后无剩余行（请确认 OOF 文件覆盖完整 train 且 val 为其子集）"
            )
        train_oof_bundle = {
            "seq_ids": np.array([s_all[i] for i in range(len(s_all)) if mask[i]], dtype=object),
            "p_esm": np.asarray(raw["p_esm"], dtype=np.float32)[mask],
            "p_gearnet": np.asarray(raw["p_gearnet"], dtype=np.float32)[mask],
            "y": np.asarray(raw["y"], dtype=np.int64)[mask],
        }
        if "p_saprot" in raw.files:
            train_oof_bundle["p_saprot"] = np.asarray(raw["p_saprot"], dtype=np.float32)[mask]
        y_oof = train_oof_bundle["y"].astype(np.int64)
    also_bundle: Optional[Dict[str, np.ndarray]] = None
    y_also: Optional[np.ndarray] = None
    if also_npz_p is not None:
        also_bundle = load_bundle(also_npz_p.resolve())
        y_also = also_bundle["y"].astype(np.int64)
    if args.holdout_npz is not None:
        test = load_bundle(args.holdout_npz.resolve())
    else:
        test = load_bundle(args.test_npz.resolve())

    ref_bundle_opt = None
    if ref_npz_p is not None:
        ref_bundle_opt = load_bundle(ref_npz_p.resolve())

    def _npz_layout_same(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], an: str, bn: str) -> None:
        if bundle_has_p_saprot(a) != bundle_has_p_saprot(b):
            raise SystemExit(f"NPZ 支路布局不一致：{an} 与 {bn} 须同为三路（含 p_saprot）或同为两路（无 p_saprot）")

    _npz_layout_same(val, test, "val_npz", "test_npz" if args.holdout_npz is None else "holdout_npz")
    two_branch_npz = not bundle_has_p_saprot(val)
    if also_bundle is not None:
        _npz_layout_same(val, also_bundle, "val_npz", "also_holdout_npz")
    if ref_bundle_opt is not None:
        _npz_layout_same(val, ref_bundle_opt, "val_npz", "meta_val_rbf_reference_holdout_npz")
    if train_oof_bundle is not None:
        _npz_layout_same(val, train_oof_bundle, "val_npz", "train_oof_npz")

    train_rows = tc.load_split(ROOT / "data" / "train_set" / "pos_train.fasta", ROOT / "data" / "train_set" / "neg_train.fasta")
    test_rows = tc.load_split(ROOT / "data" / "test_set" / "pos_test.fasta", ROOT / "data" / "test_set" / "neg_test.fasta")
    seq_map = {sid: seq for sid, _, seq in (train_rows + test_rows)}
    train_3di = tc.load_3di_map(ROOT / "data" / "train_set" / "3di" / "train_3di.fasta")
    test_3di = tc.load_3di_map(ROOT / "data" / "test_set" / "3di" / "test_3di.fasta")
    all_3di: Dict[str, str] = {}
    all_3di.update(train_3di)
    all_3di.update(test_3di)
    if args.holdout_npz is not None:
        if args.holdout_pos_fasta is None or args.holdout_neg_fasta is None:
            raise SystemExit("使用 --holdout_npz 时必须同时指定 --holdout_pos_fasta 与 --holdout_neg_fasta（用于 k-mer / 理化 / missing）")
        ho_rows = tc.load_split(args.holdout_pos_fasta.resolve(), args.holdout_neg_fasta.resolve())
        seq_map.update({sid: seq for sid, _, seq in ho_rows})
        if args.holdout_3di_fasta is not None:
            all_3di.update(tc.load_3di_map(args.holdout_3di_fasta.resolve()))
    if also_bundle is not None:
        ao_rows = tc.load_split(args.also_holdout_pos_fasta.resolve(), args.also_holdout_neg_fasta.resolve())
        seq_map.update({sid: seq for sid, _, seq in ao_rows})
        if getattr(args, "also_holdout_3di_fasta", None) is not None:
            all_3di.update(tc.load_3di_map(args.also_holdout_3di_fasta.resolve()))
    if ref_npz_p is not None:
        rr = tc.load_split(
            args.meta_val_rbf_reference_pos_fasta.resolve(),
            args.meta_val_rbf_reference_neg_fasta.resolve(),
        )
        seq_map.update({sid: seq for sid, _, seq in rr})
        if getattr(args, "meta_val_rbf_reference_3di_fasta", None) is not None:
            all_3di.update(tc.load_3di_map(args.meta_val_rbf_reference_3di_fasta.resolve()))

    plddt_map = load_plddt_map(args.plddt_csv.resolve() if args.plddt_csv else None)
    val_sids = [str(s) for s in val["seq_ids"].tolist()]
    hit_vals = [plddt_map[sid] for sid in val_sids if sid in plddt_map]
    if hit_vals:
        plddt_fill = float(np.median(hit_vals))
    else:
        plddt_fill = 70.0

    pca = PCA(n_components=int(args.kmer_pca_dim), random_state=2026)
    inc_had = not bool(getattr(args, "meta_drop_hadamard", False))
    Xv = build_X(
        val,
        seq_map,
        all_3di,
        pca,
        fit_pca=True,
        fig_extra=bool(args.fig_extra),
        plddt_map=plddt_map,
        plddt_fill=plddt_fill,
        include_prob_hadamard_pairs=inc_had,
    )
    Xt = build_X(
        test,
        seq_map,
        all_3di,
        pca,
        fit_pca=False,
        fig_extra=bool(args.fig_extra),
        plddt_map=plddt_map,
        plddt_fill=plddt_fill,
        include_prob_hadamard_pairs=inc_had,
    )

    X_ref: Optional[np.ndarray] = None
    sids_ref: List[str] = []
    if ref_bundle_opt is not None:
        sids_ref = [str(s) for s in ref_bundle_opt["seq_ids"].tolist()]
        X_ref = build_X(
            ref_bundle_opt,
            seq_map,
            all_3di,
            pca,
            fit_pca=False,
            fig_extra=bool(args.fig_extra),
            plddt_map=plddt_map,
            plddt_fill=plddt_fill,
            include_prob_hadamard_pairs=inc_had,
        )

    sids_val = [str(s) for s in val["seq_ids"].tolist()]
    sids_te = [str(s) for s in test["seq_ids"].tolist()]
    sids_also: List[str] = (
        [str(s) for s in also_bundle["seq_ids"].tolist()] if also_bundle is not None else []
    )
    X_also: Optional[np.ndarray] = None
    if also_bundle is not None:
        X_also = build_X(
            also_bundle,
            seq_map,
            all_3di,
            pca,
            fit_pca=False,
            fig_extra=bool(args.fig_extra),
            plddt_map=plddt_map,
            plddt_fill=plddt_fill,
            include_prob_hadamard_pairs=inc_had,
        )
    use_train_oof = train_oof_bundle is not None
    X_oof: Optional[np.ndarray] = None
    sids_oof: List[str] = []
    if use_train_oof:
        assert train_oof_bundle is not None and y_oof is not None
        sids_oof = [str(s) for s in train_oof_bundle["seq_ids"].tolist()]
        X_oof = build_X(
            train_oof_bundle,
            seq_map,
            all_3di,
            pca,
            fit_pca=False,
            fig_extra=bool(args.fig_extra),
            plddt_map=plddt_map,
            plddt_fill=plddt_fill,
            include_prob_hadamard_pairs=inc_had,
        )
    func_metas: List[Dict[str, object]] = []
    feat_mats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    feat_mats_oof: Dict[str, np.ndarray] = {}
    feat_mats_also: Dict[str, np.ndarray] = {}
    feat_mats_ref: Dict[str, np.ndarray] = {}
    feat_blocks = [
        ("go", args.feat_go_csv),
        ("ppi", args.feat_ppi_csv),
        ("pssm", args.feat_pssm_csv),
        ("deeploc", args.feat_deeploc_csv),
        ("binding", args.feat_binding_csv),
    ]
    pca_fdim = int(args.feat_auto_pca_dim)
    for tag, fp in feat_blocks:
        if fp is None:
            continue
        if use_train_oof:
            V, T, O, meta = align_csv_val_test_oof(
                sids_val, sids_te, sids_oof, fp.resolve() if fp else None, auto_pca_dim=pca_fdim
            )
        else:
            V, T, meta = align_csv_to_splits(
                sids_val, sids_te, fp.resolve() if fp else None, auto_pca_dim=pca_fdim
            )
            O = None
        if V.shape[1] == 0:
            continue
        meta["tag"] = tag
        func_metas.append(meta)
        feat_mats[tag] = (V.astype(np.float32), T.astype(np.float32))
        Xv = np.concatenate([Xv, V], axis=1)
        Xt = np.concatenate([Xt, T], axis=1)
        if use_train_oof and X_oof is not None and O is not None:
            Of = O.astype(np.float32)
            X_oof = np.concatenate([X_oof, Of], axis=1)
            feat_mats_oof[tag] = Of
        if also_bundle is not None and X_also is not None:
            _va, A, _ma = align_csv_to_splits(
                sids_val, sids_also, fp.resolve() if fp else None, auto_pca_dim=pca_fdim
            )
            Af = A.astype(np.float32)
            feat_mats_also[tag] = Af
            X_also = np.concatenate([X_also, Af], axis=1)
        if ref_bundle_opt is not None:
            _v2, R, _meta_r = align_csv_to_splits(
                sids_val, sids_ref, fp.resolve() if fp else None, auto_pca_dim=pca_fdim
            )
            Rf = R.astype(np.float32)
            X_ref = np.concatenate([X_ref, Rf], axis=1)
            feat_mats_ref[tag] = Rf

    if args.meta_cross_features:
        if args.feat_go_csv is None or args.feat_ppi_csv is None:
            raise SystemExit("--meta_cross_features 需要同时指定 --feat_go_csv 与 --feat_ppi_csv")
        if "go" not in feat_mats or "ppi" not in feat_mats:
            raise SystemExit("--meta_cross_features 需要 go 与 ppi 功能块均成功对齐且维数>0")
        Vgo, Tgo = feat_mats["go"]
        Vppi, Tppi = feat_mats["ppi"]
        pe_v = val["p_esm"].astype(np.float64).reshape(-1)
        ppi_mean_v = np.mean(Vppi.astype(np.float64), axis=1)
        go_pc1_v = Vgo[:, 0].astype(np.float64)
        if two_branch_npz:
            pg_v = val["p_gearnet"].astype(np.float64).reshape(-1)
            cross_v = np.column_stack([pe_v * ppi_mean_v, pg_v * go_pc1_v]).astype(np.float32)
        else:
            ps_v = val["p_saprot"].astype(np.float64).reshape(-1)
            cross_v = np.column_stack([pe_v * ppi_mean_v, ps_v * go_pc1_v]).astype(np.float32)
        pe_t = test["p_esm"].astype(np.float64).reshape(-1)
        ppi_mean_t = np.mean(Tppi.astype(np.float64), axis=1)
        go_pc1_t = Tgo[:, 0].astype(np.float64)
        if two_branch_npz:
            pg_t = test["p_gearnet"].astype(np.float64).reshape(-1)
            cross_t = np.column_stack([pe_t * ppi_mean_t, pg_t * go_pc1_t]).astype(np.float32)
        else:
            ps_t = test["p_saprot"].astype(np.float64).reshape(-1)
            cross_t = np.column_stack([pe_t * ppi_mean_t, ps_t * go_pc1_t]).astype(np.float32)
        Xv = np.concatenate([Xv, cross_v], axis=1)
        Xt = np.concatenate([Xt, cross_t], axis=1)
        if use_train_oof and X_oof is not None:
            if "go" not in feat_mats_oof or "ppi" not in feat_mats_oof:
                raise SystemExit(
                    "内部错误：--meta_cross_features 与 --train_oof_npz 同开时需要在功能 CSV 循环中得到 go/ppi 的 OOF 对齐矩阵"
                )
            Ogo = feat_mats_oof["go"]
            Oppi = feat_mats_oof["ppi"]
            pe_o = train_oof_bundle["p_esm"].astype(np.float64).reshape(-1)
            ppi_mean_o = np.mean(Oppi.astype(np.float64), axis=1)
            go_pc1_o = Ogo[:, 0].astype(np.float64)
            if two_branch_npz:
                pg_o = train_oof_bundle["p_gearnet"].astype(np.float64).reshape(-1)
                cross_o = np.column_stack([pe_o * ppi_mean_o, pg_o * go_pc1_o]).astype(np.float32)
            else:
                ps_o = train_oof_bundle["p_saprot"].astype(np.float64).reshape(-1)
                cross_o = np.column_stack([pe_o * ppi_mean_o, ps_o * go_pc1_o]).astype(np.float32)
            X_oof = np.concatenate([X_oof, cross_o], axis=1)
        if also_bundle is not None and X_also is not None:
            if "go" not in feat_mats_also or "ppi" not in feat_mats_also:
                raise SystemExit("内部错误：meta_cross 需要 also 侧 go/ppi 对齐块")
            Ago = feat_mats_also["go"]
            Appi = feat_mats_also["ppi"]
            pe_a = also_bundle["p_esm"].astype(np.float64).reshape(-1)
            ppi_mean_a = np.mean(Appi.astype(np.float64), axis=1)
            go_pc1_a = Ago[:, 0].astype(np.float64)
            if two_branch_npz:
                pg_a = also_bundle["p_gearnet"].astype(np.float64).reshape(-1)
                cross_a = np.column_stack([pe_a * ppi_mean_a, pg_a * go_pc1_a]).astype(np.float32)
            else:
                ps_a = also_bundle["p_saprot"].astype(np.float64).reshape(-1)
                cross_a = np.column_stack([pe_a * ppi_mean_a, ps_a * go_pc1_a]).astype(np.float32)
            X_also = np.concatenate([X_also, cross_a], axis=1)
        if ref_bundle_opt is not None:
            Rgo = feat_mats_ref["go"]
            Rppi = feat_mats_ref["ppi"]
            pe_r = ref_bundle_opt["p_esm"].astype(np.float64).reshape(-1)
            ppi_mean_r = np.mean(Rppi.astype(np.float64), axis=1)
            go_pc1_r = Rgo[:, 0].astype(np.float64)
            if two_branch_npz:
                pg_r = ref_bundle_opt["p_gearnet"].astype(np.float64).reshape(-1)
                cross_r = np.column_stack([pe_r * ppi_mean_r, pg_r * go_pc1_r]).astype(np.float32)
            else:
                ps_r = ref_bundle_opt["p_saprot"].astype(np.float64).reshape(-1)
                cross_r = np.column_stack([pe_r * ppi_mean_r, ps_r * go_pc1_r]).astype(np.float32)
            X_ref = np.concatenate([X_ref, cross_r], axis=1)

    motif_dim = 0
    if args.afp_local_motif:
        Mv = np.stack([local_afp_motif_feats(seq_map.get(sid, "")) for sid in sids_val], axis=0)
        Mt = np.stack([local_afp_motif_feats(seq_map.get(sid, "")) for sid in sids_te], axis=0)
        Xv = np.concatenate([Xv, Mv.astype(np.float32)], axis=1)
        Xt = np.concatenate([Xt, Mt.astype(np.float32)], axis=1)
        if use_train_oof and X_oof is not None:
            Mo = np.stack([local_afp_motif_feats(seq_map.get(sid, "")) for sid in sids_oof], axis=0)
            X_oof = np.concatenate([X_oof, Mo.astype(np.float32)], axis=1)
        if also_bundle is not None and X_also is not None:
            Ma = np.stack([local_afp_motif_feats(seq_map.get(sid, "")) for sid in sids_also], axis=0)
            X_also = np.concatenate([X_also, Ma.astype(np.float32)], axis=1)
        motif_dim = int(Mv.shape[1])

    seq_len_dim = 0
    val_median_aa_len_for_meta = 1.0
    if args.meta_seq_len:
        lens_v = [
            len("".join(c for c in seq_map.get(sid, "").upper() if c in AA_INDEX))
            for sid in sids_val
        ]
        val_median_aa_len_for_meta = float(np.median(lens_v)) if lens_v else 1.0
        val_median_aa_len_for_meta = max(val_median_aa_len_for_meta, 1.0)
        pe_vl = val["p_esm"].astype(np.float64).reshape(-1)
        pg_vl = val["p_gearnet"].astype(np.float64).reshape(-1)
        pe_tl = test["p_esm"].astype(np.float64).reshape(-1)
        pg_tl = test["p_gearnet"].astype(np.float64).reshape(-1)
        Sv = np.stack(
            [
                meta_seq_len_gate_row(seq_map.get(sid, ""), float(pe_vl[i]), float(pg_vl[i]), val_median_aa_len_for_meta)
                for i, sid in enumerate(sids_val)
            ],
            axis=0,
        )
        St = np.stack(
            [
                meta_seq_len_gate_row(seq_map.get(sid, ""), float(pe_tl[i]), float(pg_tl[i]), val_median_aa_len_for_meta)
                for i, sid in enumerate(sids_te)
            ],
            axis=0,
        )
        Xv = np.concatenate([Xv, Sv.astype(np.float32)], axis=1)
        Xt = np.concatenate([Xt, St.astype(np.float32)], axis=1)
        if use_train_oof and X_oof is not None:
            pe_ol = train_oof_bundle["p_esm"].astype(np.float64).reshape(-1)
            pg_ol = train_oof_bundle["p_gearnet"].astype(np.float64).reshape(-1)
            So = np.stack(
                [
                    meta_seq_len_gate_row(seq_map.get(sid, ""), float(pe_ol[i]), float(pg_ol[i]), val_median_aa_len_for_meta)
                    for i, sid in enumerate(sids_oof)
                ],
                axis=0,
            )
            X_oof = np.concatenate([X_oof, So.astype(np.float32)], axis=1)
        if also_bundle is not None and X_also is not None:
            pe_al = also_bundle["p_esm"].astype(np.float64).reshape(-1)
            pg_al = also_bundle["p_gearnet"].astype(np.float64).reshape(-1)
            Sa = np.stack(
                [
                    meta_seq_len_gate_row(
                        seq_map.get(sid, ""), float(pe_al[i]), float(pg_al[i]), val_median_aa_len_for_meta
                    )
                    for i, sid in enumerate(sids_also)
                ],
                axis=0,
            )
            X_also = np.concatenate([X_also, Sa.astype(np.float32)], axis=1)
        seq_len_dim = int(Sv.shape[1])

    robust_dim = 0
    if args.robust_ood_meta:
        if not args.fig_extra:
            raise SystemExit("--robust_ood_meta 需要 --fig_extra 或 --four_panel（以包含 pLDDT 列）")
        pe_v = val["p_esm"].astype(np.float64).reshape(-1)
        pg_v = val["p_gearnet"].astype(np.float64).reshape(-1)
        pe_t = test["p_esm"].astype(np.float64).reshape(-1)
        pg_t = test["p_gearnet"].astype(np.float64).reshape(-1)
        pl_v = _plddt_values(sids_val, plddt_map, plddt_fill).astype(np.float64)
        pl_t = _plddt_values(sids_te, plddt_map, plddt_fill).astype(np.float64)
        sc = max(float(args.robust_plddt_scale), 1e-6)
        gate_v = _sigmoid_np((pl_v - float(args.robust_plddt_center)) / sc).astype(np.float32)
        gate_t = _sigmoid_np((pl_t - float(args.robust_plddt_center)) / sc).astype(np.float32)
        if two_branch_npz:
            blk_v = [
                gate_v.reshape(-1, 1),
                (pg_v * gate_v).reshape(-1, 1).astype(np.float32),
                (pe_v * (1.0 - gate_v.astype(np.float64))).reshape(-1, 1).astype(np.float32),
            ]
            blk_t = [
                gate_t.reshape(-1, 1),
                (pg_t * gate_t).reshape(-1, 1).astype(np.float32),
                (pe_t * (1.0 - gate_t.astype(np.float64))).reshape(-1, 1).astype(np.float32),
            ]
            robust_dim = 3
        else:
            ps_v = val["p_saprot"].astype(np.float64).reshape(-1)
            ps_t = test["p_saprot"].astype(np.float64).reshape(-1)
            blk_v = [
                gate_v.reshape(-1, 1),
                (pg_v * gate_v).reshape(-1, 1).astype(np.float32),
                (ps_v * gate_v).reshape(-1, 1).astype(np.float32),
                (pe_v * (1.0 - gate_v.astype(np.float64))).reshape(-1, 1).astype(np.float32),
            ]
            blk_t = [
                gate_t.reshape(-1, 1),
                (pg_t * gate_t).reshape(-1, 1).astype(np.float32),
                (ps_t * gate_t).reshape(-1, 1).astype(np.float32),
                (pe_t * (1.0 - gate_t.astype(np.float64))).reshape(-1, 1).astype(np.float32),
            ]
            robust_dim = 4
        if args.feat_ppi_csv is not None:
            blk_v.append(ppi_raw_row_stats_matrix(sids_val, args.feat_ppi_csv.resolve()))
            blk_t.append(ppi_raw_row_stats_matrix(sids_te, args.feat_ppi_csv.resolve()))
            robust_dim += 3
        Xv = np.concatenate([Xv] + blk_v, axis=1)
        Xt = np.concatenate([Xt] + blk_t, axis=1)
        if use_train_oof and X_oof is not None:
            pe_o = train_oof_bundle["p_esm"].astype(np.float64).reshape(-1)
            pg_o = train_oof_bundle["p_gearnet"].astype(np.float64).reshape(-1)
            pl_o = _plddt_values(sids_oof, plddt_map, plddt_fill).astype(np.float64)
            gate_o = _sigmoid_np((pl_o - float(args.robust_plddt_center)) / sc).astype(np.float32)
            if two_branch_npz:
                blk_o = [
                    gate_o.reshape(-1, 1),
                    (pg_o * gate_o).reshape(-1, 1).astype(np.float32),
                    (pe_o * (1.0 - gate_o.astype(np.float64))).reshape(-1, 1).astype(np.float32),
                ]
            else:
                ps_o = train_oof_bundle["p_saprot"].astype(np.float64).reshape(-1)
                blk_o = [
                    gate_o.reshape(-1, 1),
                    (pg_o * gate_o).reshape(-1, 1).astype(np.float32),
                    (ps_o * gate_o).reshape(-1, 1).astype(np.float32),
                    (pe_o * (1.0 - gate_o.astype(np.float64))).reshape(-1, 1).astype(np.float32),
                ]
            if args.feat_ppi_csv is not None:
                blk_o.append(ppi_raw_row_stats_matrix(sids_oof, args.feat_ppi_csv.resolve()))
            X_oof = np.concatenate([X_oof] + blk_o, axis=1)
        if also_bundle is not None and X_also is not None:
            pe_a = also_bundle["p_esm"].astype(np.float64).reshape(-1)
            pg_a = also_bundle["p_gearnet"].astype(np.float64).reshape(-1)
            pl_a = _plddt_values(sids_also, plddt_map, plddt_fill).astype(np.float64)
            gate_a = _sigmoid_np((pl_a - float(args.robust_plddt_center)) / sc).astype(np.float32)
            if two_branch_npz:
                blk_a = [
                    gate_a.reshape(-1, 1),
                    (pg_a * gate_a).reshape(-1, 1).astype(np.float32),
                    (pe_a * (1.0 - gate_a.astype(np.float64))).reshape(-1, 1).astype(np.float32),
                ]
            else:
                ps_a = also_bundle["p_saprot"].astype(np.float64).reshape(-1)
                blk_a = [
                    gate_a.reshape(-1, 1),
                    (pg_a * gate_a).reshape(-1, 1).astype(np.float32),
                    (ps_a * gate_a).reshape(-1, 1).astype(np.float32),
                    (pe_a * (1.0 - gate_a.astype(np.float64))).reshape(-1, 1).astype(np.float32),
                ]
            if args.feat_ppi_csv is not None:
                blk_a.append(ppi_raw_row_stats_matrix(sids_also, args.feat_ppi_csv.resolve()))
            X_also = np.concatenate([X_also] + blk_a, axis=1)
        if ref_bundle_opt is not None:
            pe_r = ref_bundle_opt["p_esm"].astype(np.float64).reshape(-1)
            pg_r = ref_bundle_opt["p_gearnet"].astype(np.float64).reshape(-1)
            pl_r = _plddt_values(sids_ref, plddt_map, plddt_fill).astype(np.float64)
            gate_r = _sigmoid_np((pl_r - float(args.robust_plddt_center)) / sc).astype(np.float32)
            if two_branch_npz:
                blk_r = [
                    gate_r.reshape(-1, 1),
                    (pg_r * gate_r).reshape(-1, 1).astype(np.float32),
                    (pe_r * (1.0 - gate_r.astype(np.float64))).reshape(-1, 1).astype(np.float32),
                ]
            else:
                ps_r = ref_bundle_opt["p_saprot"].astype(np.float64).reshape(-1)
                blk_r = [
                    gate_r.reshape(-1, 1),
                    (pg_r * gate_r).reshape(-1, 1).astype(np.float32),
                    (ps_r * gate_r).reshape(-1, 1).astype(np.float32),
                    (pe_r * (1.0 - gate_r.astype(np.float64))).reshape(-1, 1).astype(np.float32),
                ]
            if args.feat_ppi_csv is not None:
                blk_r.append(ppi_raw_row_stats_matrix(sids_ref, args.feat_ppi_csv.resolve()))
            X_ref = np.concatenate([X_ref] + blk_r, axis=1)

    yv = val["y"].astype(np.int64)
    yt = test["y"].astype(np.int64)

    n_base_probs_meta = 2 if two_branch_npz else 3
    hadamard_meta_dim = (
        0
        if getattr(args, "meta_drop_hadamard", False)
        else (1 if two_branch_npz else 3)
    )
    fig_disagreement_cols = (8 if two_branch_npz else 12) if args.fig_extra else 0
    miss_col = n_base_probs_meta + hadamard_meta_dim + int(args.kmer_pca_dim)
    if Xv.shape[1] <= miss_col:
        raise SystemExit(f"特征维 {Xv.shape[1]} 不足 missing 列索引 {miss_col}（kmer_pca_dim={args.kmer_pca_dim}）")

    col_blocks = meta_lr_column_blocks(
        kmer_pca_dim=int(args.kmer_pca_dim),
        fig_extra=bool(args.fig_extra),
        func_metas=func_metas,
        meta_cross=bool(args.meta_cross_features),
        motif_dim=int(motif_dim),
        seq_len_dim=int(seq_len_dim),
        robust_dim=int(robust_dim),
        include_hadamard=not bool(getattr(args, "meta_drop_hadamard", False)),
        n_base_probs=int(n_base_probs_meta),
        hadamard_width=int(hadamard_meta_dim),
        fig_disagreement_cols=int((8 if two_branch_npz else 12) if args.fig_extra else 12),
    )
    expect_d_cb = col_blocks[-1][2] if col_blocks else 0
    if int(Xv.shape[1]) != expect_d_cb:
        raise SystemExit(f"内部错误：列块与特征维不一致 Xv.shape[1]={Xv.shape[1]} 期望={expect_d_cb}")
    if ref_bundle_opt is not None and int(X_ref.shape[1]) != int(Xv.shape[1]):
        raise SystemExit(f"内部错误：reference Meta 特征维 {X_ref.shape[1]} 与 Xv {Xv.shape[1]} 不一致")
    if also_bundle is not None and X_also is not None and int(X_also.shape[1]) != int(Xt.shape[1]):
        raise SystemExit(
            f"内部错误：also Meta 特征维 {X_also.shape[1]} 与 Xt {Xt.shape[1]} 不一致（请检查 also 与 test 的特征分支是否一致）"
        )
    if use_train_oof and X_oof is not None and int(X_oof.shape[1]) != int(Xv.shape[1]):
        raise SystemExit(
            f"内部错误：train OOF Meta 特征维 {X_oof.shape[1]} 与 Xv {Xv.shape[1]} 不一致"
        )

    miss_v = Xv[:, miss_col].astype(np.float64).copy()
    miss_t = Xt[:, miss_col].astype(np.float64).copy()

    scaler: StandardScaler | None = None
    if args.scale_meta:
        scaler = StandardScaler()
        Xv_fit = scaler.fit_transform(Xv)
        Xt_fit = scaler.transform(Xt)
    else:
        Xv_fit, Xt_fit = Xv, Xt
    X_also_fit: Optional[np.ndarray] = None
    if X_also is not None:
        X_also_fit = scaler.transform(X_also) if scaler is not None else X_also
    X_ref_fit: Optional[np.ndarray] = None
    if ref_bundle_opt is not None:
        X_ref_fit = scaler.transform(X_ref) if scaler is not None else np.asarray(X_ref, dtype=np.float32)
    X_oof_fit: Optional[np.ndarray] = None
    if X_oof is not None:
        X_oof_fit = scaler.transform(X_oof) if scaler is not None else np.asarray(X_oof, dtype=np.float32)

    X_train_meta = Xv_fit
    y_train_meta = yv
    if X_oof_fit is not None and y_oof is not None:
        X_train_meta = np.ascontiguousarray(np.vstack([Xv_fit, X_oof_fit.astype(np.float32)]))
        y_train_meta = np.concatenate([yv, y_oof], axis=0)
    meta_cls = str(args.meta_classifier or "lr").lower()
    xgb_meta_clfs: List[object] = []
    val_rbf_w: Optional[np.ndarray] = None
    g_rbf = float(getattr(args, "meta_val_reweight_holdout_rbf_gamma", 0.0) or 0.0)
    if g_rbf > 0.0:
        if args.holdout_npz is not None:
            canvas = Xt_fit
        elif X_ref_fit is not None:
            canvas = X_ref_fit
        else:
            raise SystemExit(
                "--meta-val-reweight-holdout-rbf-gamma>0 时需 --holdout_npz，"
                "或同时指定 --meta-val-rbf-reference-holdout-npz 与参考 FASTA"
            )
        base_seed = int(getattr(args, "meta_xgb_bootstrap_seed", 2026) or 2026)
        n_avg = int(getattr(args, "meta_val_reweight_holdout_rbf_avg_seeds", 1) or 1)
        if n_avg < 1:
            raise SystemExit("--meta-val-reweight-holdout-rbf-avg-seeds 必须 >=1")
        need_rbf_w = (
            meta_cls == "xgboost"
            or bool(getattr(args, "meta_lr_use_rbf_sample_weight", False))
            or (meta_cls == "lr" and aux_lr_xgb_blend)
        )
        acc_w: Optional[np.ndarray] = None
        if need_rbf_w:
            for k in range(n_avg):
                rng_w = np.random.RandomState(base_seed + k)
                w_k = _holdout_rbf_val_sample_weights(
                    Xv_fit,
                    canvas,
                    gamma=g_rbf,
                    alpha=float(getattr(args, "meta_val_reweight_holdout_alpha", 0.5) or 0.5),
                    max_ho=int(getattr(args, "meta_val_reweight_holdout_max_rows", 512) or 512),
                    rng=rng_w,
                )
                acc_w = w_k.copy() if acc_w is None else (acc_w + w_k)
            assert acc_w is not None
            val_rbf_w = (acc_w / float(n_avg)).astype(np.float64)
            val_rbf_w = val_rbf_w / (float(np.mean(val_rbf_w)) + 1e-12)
    fit_rbf_lr: Optional[np.ndarray] = None
    if val_rbf_w is not None:
        vr0 = np.asarray(val_rbf_w, dtype=np.float64).reshape(-1)
        if X_oof_fit is not None:
            wc = np.concatenate([vr0, np.ones(int(X_oof_fit.shape[0]), dtype=np.float64)])
            fit_rbf_lr = (wc / (float(np.mean(wc)) + 1e-12)).astype(np.float64)
        else:
            fit_rbf_lr = vr0
    if args.meta_moe_missing and meta_cls != "lr":
        raise SystemExit("--meta_moe_missing 仅支持 --meta_classifier lr（勿与 xgboost 同开）")

    if args.meta_moe_missing:
        low, high = float(args.moe_gate_low), float(args.moe_gate_high)
        if high <= low + 1e-9:
            raise SystemExit("--moe_gate_high 必须大于 --moe_gate_low")
        boost = float(args.moe_expert_boost)
        w_r = np.where(miss_v < low, boost, 1.0).astype(np.float64)
        w_p = np.where(miss_v > high, boost, 1.0).astype(np.float64)
        if val_rbf_w is not None and bool(getattr(args, "meta_lr_use_rbf_sample_weight", True)):
            vr = np.asarray(val_rbf_w, dtype=np.float64).reshape(-1)
            w_r = (w_r * vr) / (float(np.mean(w_r * vr)) + 1e-12)
            w_p = (w_p * vr) / (float(np.mean(w_p * vr)) + 1e-12)
        clf_rich = _make_meta_clf(args.stacking_penalty, args.stacking_C)
        clf_poor = _make_meta_clf(args.stacking_penalty, args.stacking_C)
        clf_rich.fit(Xv_fit, yv, sample_weight=w_r)
        clf_poor.fit(Xv_fit, yv, sample_weight=w_p)
        gv = np.clip((miss_v - low) / (high - low), 0.0, 1.0)
        gt = np.clip((miss_t - low) / (high - low), 0.0, 1.0)
        pr_v = clf_rich.predict_proba(Xv_fit)[:, 1].astype(np.float64)
        po_v = clf_poor.predict_proba(Xv_fit)[:, 1].astype(np.float64)
        pr_t = clf_rich.predict_proba(Xt_fit)[:, 1].astype(np.float64)
        po_t = clf_poor.predict_proba(Xt_fit)[:, 1].astype(np.float64)
        prob_va = (1.0 - gv) * pr_v + gv * po_v
        prob_te = (1.0 - gt) * pr_t + gt * po_t
        clf = clf_rich
    elif meta_cls == "xgboost":
        if X_oof_fit is not None and y_oof is not None:
            xgb_meta_clfs, prob_va, prob_te = _fit_meta_xgb_models(
                args,
                Xv_fit=Xv_fit,
                yv=yv,
                Xt_fit=Xt_fit,
                miss_v=miss_v,
                extra_val_weights=val_rbf_w,
                X_augment_fit=X_oof_fit,
                y_augment=y_oof,
                miss_augment=X_oof[:, miss_col].astype(np.float64).copy(),
            )
        else:
            xgb_meta_clfs, prob_va, prob_te = _fit_meta_xgb_models(
                args,
                Xv_fit=Xv_fit,
                yv=yv,
                Xt_fit=Xt_fit,
                miss_v=miss_v,
                extra_val_weights=val_rbf_w,
            )
        clf = xgb_meta_clfs[-1]
    else:
        clf = _make_meta_clf(args.stacking_penalty, args.stacking_C)
        fit_lr: Dict[str, object] = {}
        n_lm = int(X_train_meta.shape[0])
        n_val_rows = int(Xv_fit.shape[0])
        w_lm = np.ones(n_lm, dtype=np.float64)
        if fit_rbf_lr is not None and bool(getattr(args, "meta_lr_use_rbf_sample_weight", True)):
            w_lm = np.asarray(fit_rbf_lr, dtype=np.float64).reshape(-1).copy()
        oof_rw_lm = float(getattr(args, "meta_train_oof_augment_row_weight", 1.0) or 1.0)
        if (
            X_oof_fit is not None
            and y_oof is not None
            and n_lm > n_val_rows
            and abs(oof_rw_lm - 1.0) > 1e-15
        ):
            w_lm[n_val_rows:] *= oof_rw_lm
        need_sw = (fit_rbf_lr is not None and bool(getattr(args, "meta_lr_use_rbf_sample_weight", True))) or (
            X_oof_fit is not None
            and y_oof is not None
            and n_lm > n_val_rows
            and abs(oof_rw_lm - 1.0) > 1e-15
        )
        if need_sw:
            w_lm = w_lm / (float(np.mean(w_lm)) + 1e-12)
            fit_lr["sample_weight"] = w_lm
        clf.fit(X_train_meta, y_train_meta, **fit_lr)
        prob_te = clf.predict_proba(Xt_fit)[:, 1]
        prob_va = clf.predict_proba(Xv_fit)[:, 1]

    meta_lr_aux_xgb_blend_report: Optional[Dict[str, object]] = None
    meta_lr_fuse_primary_after_aux_blend_report: Optional[Dict[str, object]] = None
    aux_xgb_clfs: List[object] = []
    if meta_cls == "lr" and aux_lr_xgb_blend:
        prob_va_lr_meta_pre_aux: Optional[np.ndarray] = None
        prob_te_lr_meta_pre_aux: Optional[np.ndarray] = None
        if fuse_lr_after_aux:
            prob_va_lr_meta_pre_aux = np.asarray(prob_va, dtype=np.float64).copy()
            prob_te_lr_meta_pre_aux = np.asarray(prob_te, dtype=np.float64).copy()
        if X_oof_fit is not None and y_oof is not None:
            aux_xgb_clfs, prob_va_xgb_aux, prob_te_xgb_aux = _fit_meta_xgb_models(
                args,
                Xv_fit=Xv_fit,
                yv=yv,
                Xt_fit=Xt_fit,
                miss_v=miss_v,
                extra_val_weights=val_rbf_w,
                X_augment_fit=X_oof_fit,
                y_augment=y_oof,
                miss_augment=X_oof[:, miss_col].astype(np.float64).copy(),
            )
        else:
            aux_xgb_clfs, prob_va_xgb_aux, prob_te_xgb_aux = _fit_meta_xgb_models(
                args,
                Xv_fit=Xv_fit,
                yv=yv,
                Xt_fit=Xt_fit,
                miss_v=miss_v,
                extra_val_weights=val_rbf_w,
            )
        prob_va, prob_te, meta_lr_aux_xgb_blend_report = apply_meta_aux_prob_blend_tune(
            prob_va_primary=prob_va,
            prob_te_primary=prob_te,
            prob_va_aux=prob_va_xgb_aux,
            prob_te_aux=prob_te_xgb_aux,
            yv=yv,
            w_max=float(getattr(args, "meta_lr_aux_xgb_blend_w_max", 1.0)),
            n_grid=int(getattr(args, "meta_lr_aux_xgb_blend_grid", 51)),
            mcc_slack=float(getattr(args, "meta_lr_aux_xgb_blend_mcc_slack", 0.003)),
            val_select_penalty_lambda=float(getattr(args, "meta_lr_aux_xgb_blend_val_penalty_lambda", 0.0) or 0.0),
        )
        if fuse_lr_after_aux and prob_va_lr_meta_pre_aux is not None and prob_te_lr_meta_pre_aux is not None:
            prob_va, prob_te, meta_lr_fuse_primary_after_aux_blend_report = apply_meta_fuse_primary_after_aux_blend_tune(
                prob_va_blend=prob_va,
                prob_te_blend=prob_te,
                prob_va_lr_meta=prob_va_lr_meta_pre_aux,
                prob_te_lr_meta=prob_te_lr_meta_pre_aux,
                yv=yv,
                a_max=float(getattr(args, "meta_lr_fuse_primary_after_aux_blend_a_max", 0.5) or 0.5),
                n_grid=int(getattr(args, "meta_lr_fuse_primary_after_aux_blend_grid", 41) or 41),
                mcc_slack=float(getattr(args, "meta_lr_fuse_primary_after_aux_blend_mcc_slack", 0.0) or 0.0),
            )

    mean3_blend_report: Optional[Dict[str, object]] = None
    if getattr(args, "meta_mean3_blend", False):
        prob_va, prob_te, mean3_blend_report = apply_meta_mean3_blend_tune(
            prob_va=prob_va,
            prob_te=prob_te,
            yv=yv,
            val_bundle=val,
            test_bundle=test,
            w_max=float(args.meta_mean3_blend_w_max),
            n_grid=int(args.meta_mean3_blend_grid),
            mcc_slack=float(args.meta_mean3_blend_mcc_slack),
        )

    platt_report: Optional[Dict[str, object]] = None
    if getattr(args, "platt_calibrate", False):
        prob_va_uncal = np.asarray(prob_va, dtype=np.float64).copy()
        prob_te_uncal = np.asarray(prob_te, dtype=np.float64).copy()
        prob_va, prob_te, platt_report = platt_fit_apply(
            prob_va_uncal=prob_va_uncal,
            prob_te_uncal=prob_te_uncal,
            yv=yv,
            yt=yt,
            clf=clf,
            Xv_fit=Xv_fit,
            Xt_fit=Xt_fit,
            meta_cls=meta_cls,
            meta_moe_missing=bool(args.meta_moe_missing),
        )

    shap_bundle_dir = getattr(args, "export_meta_shap_bundle", None)
    if shap_bundle_dir is not None:
        if meta_cls != "xgboost" or not xgb_meta_clfs:
            print(
                "WARN: --export-meta-shap-bundle 仅适用于 meta_classifier=xgboost 且已训练 Meta-XGB；跳过导出。",
                flush=True,
            )
        else:
            import joblib

            out_shap = Path(str(shap_bundle_dir)).resolve()
            out_shap.mkdir(parents=True, exist_ok=True)
            joblib.dump(xgb_meta_clfs, out_shap / "xgb_meta_classifiers.joblib")
            joblib.dump(scaler, out_shap / "meta_scaler.joblib")
            np.save(out_shap / "X_val_fit.npy", np.asarray(Xv_fit, dtype=np.float32))
            np.save(out_shap / "y_val.npy", np.asarray(yv, dtype=np.int64))
            np.save(out_shap / "X_test_fit.npy", np.asarray(Xt_fit, dtype=np.float32))
            np.save(out_shap / "y_test.npy", np.asarray(yt, dtype=np.int64))
            nbp = int(n_base_probs_meta)
            np.save(out_shap / "test_base_probs.npy", np.asarray(Xt[:, :nbp], dtype=np.float32))
            np.save(out_shap / "val_base_probs.npy", np.asarray(Xv[:, :nbp], dtype=np.float32))
            p_uncal_te = _meta_xgb_proba_positive_mean(xgb_meta_clfs, Xt_fit).astype(np.float64)
            np.save(out_shap / "p_meta_test_uncalibrated.npy", p_uncal_te)
            blocks_json = [{"name": n, "lo": int(lo), "hi": int(hi)} for n, lo, hi in col_blocks]
            with (out_shap / "column_blocks.json").open("w", encoding="utf-8") as fp:
                json.dump(blocks_json, fp, indent=2, ensure_ascii=False)
            lab = (["p_esm", "p_gearnet"] if two_branch_npz else ["p_esm", "p_saprot", "p_gearnet"])[:nbp]
            manifest = {
                "n_base_probs": nbp,
                "base_prob_labels": lab,
                "two_branch_npz": bool(two_branch_npz),
                "scale_meta": bool(args.scale_meta),
                "note": "SHAP/贡献度针对 Meta-XGB 未校准正类概率；与 p_meta_test_uncalibrated.npy 一致。",
            }
            with (out_shap / "manifest.json").open("w", encoding="utf-8") as fp:
                json.dump(manifest, fp, indent=2, ensure_ascii=False)
            print("Wrote meta SHAP bundle under", out_shap, flush=True)

    holdout_extra_reports: Dict[str, object] = {}
    ztags = {
        x.strip().lower()
        for x in (getattr(args, "holdout_zero_feat_csv_tags", "") or "").split(",")
        if x.strip()
    }
    if args.holdout_npz is not None and ztags:
        Xt_alt = np.asarray(Xt, dtype=np.float32).copy()
        nz = 0
        for name, lo, hi in col_blocks:
            for t in ztags:
                if name.startswith(f"feat_csv_{t}"):
                    Xt_alt[:, lo:hi] = 0.0
                    nz += hi - lo
                    break
        Xt_alt_fit = scaler.transform(Xt_alt) if scaler is not None else Xt_alt
        if args.meta_moe_missing:
            pr_z = clf_rich.predict_proba(Xt_alt_fit)[:, 1].astype(np.float64)
            po_z = clf_poor.predict_proba(Xt_alt_fit)[:, 1].astype(np.float64)
            prob_te_z = (1.0 - gt) * pr_z + gt * po_z
        elif meta_cls == "xgboost":
            prob_te_z = _meta_xgb_proba_positive_mean(xgb_meta_clfs, Xt_alt_fit).astype(np.float64)
        else:
            prob_te_z = clf.predict_proba(Xt_alt_fit)[:, 1].astype(np.float64)
        holdout_extra_reports["holdout_feat_csv_zeroed_n_floats"] = int(nz)
        holdout_extra_reports["test_metrics_meta_holdout_feat_csv_zeroed_uncalibrated"] = metrics_block(
            yt, prob_te_z
        )
        if getattr(args, "platt_calibrate", False):
            holdout_extra_reports["holdout_feat_csv_zeroed_platt_note"] = (
                "此为置零后 Meta 概率，未再次套用 val 上 Platt；主 test_metrics 仍为完整特征 +（若开启）Platt。"
            )

    if args.holdout_npz is not None and bool(getattr(args, "holdout_report_base_fusions", False)):
        pe = test["p_esm"].astype(np.float64).reshape(-1)
        pg = test["p_gearnet"].astype(np.float64).reshape(-1)
        if two_branch_npz:
            pm = (pe + pg) / 2.0
            pmed = np.median(np.stack([pe, pg], axis=0), axis=0)
        else:
            ps = test["p_saprot"].astype(np.float64).reshape(-1)
            pm = (pe + ps + pg) / 3.0
            pmed = np.median(np.stack([pe, ps, pg], axis=0), axis=0)
        holdout_extra_reports["test_metrics_prob_mean3"] = metrics_block(yt, pm)
        holdout_extra_reports["test_metrics_prob_median3"] = metrics_block(yt, pmed)
        if two_branch_npz:
            holdout_extra_reports["holdout_base_fusion_note"] = (
                "NPZ 为两路（无 p_saprot）：mean3/median3 键名沿用，实为 (p_esm+p_gearnet)/2 与 median(两路)。"
            )
    elif (
        also_bundle is not None
        and y_also is not None
        and bool(getattr(args, "holdout_report_base_fusions", False))
        and bool(getattr(args, "also_val_pick_triple_fusion", False))
    ):
        pea = also_bundle["p_esm"].astype(np.float64).reshape(-1)
        pga = also_bundle["p_gearnet"].astype(np.float64).reshape(-1)
        if two_branch_npz:
            pm_a = (pea + pga) / 2.0
            pmed_a = np.median(np.stack([pea, pga], axis=0), axis=0)
        else:
            psa = also_bundle["p_saprot"].astype(np.float64).reshape(-1)
            pm_a = (pea + psa + pga) / 3.0
            pmed_a = np.median(np.stack([pea, psa, pga], axis=0), axis=0)
        holdout_extra_reports["also_metrics_prob_mean3"] = metrics_block(y_also, pm_a)
        holdout_extra_reports["also_metrics_prob_median3"] = metrics_block(y_also, pmed_a)

    prob_also_for_export: Optional[np.ndarray] = None
    prob_also_f: Optional[np.ndarray] = None
    test_metrics_also_holdout_npz: Optional[Dict[str, float]] = None
    if also_bundle is not None and X_also_fit is not None and y_also is not None:
        prob_also_f: np.ndarray
        if meta_cls == "lr" and aux_lr_xgb_blend:
            if meta_lr_aux_xgb_blend_report is None:
                raise SystemExit("内部错误：meta_lr_aux_xgb_blend_report 为空")
            wf = float(meta_lr_aux_xgb_blend_report["meta_lr_aux_xgb_blend_w"])
            p_lr_a = clf.predict_proba(X_also_fit)[:, 1].astype(np.float64)
            p_xg_a = _meta_xgb_proba_positive_mean(aux_xgb_clfs, X_also_fit).astype(np.float64)
            prob_blend_also = (1.0 - wf) * p_lr_a + wf * p_xg_a
            if meta_lr_fuse_primary_after_aux_blend_report is not None:
                af = float(
                    meta_lr_fuse_primary_after_aux_blend_report["meta_lr_fuse_primary_after_aux_blend_alpha"]
                )
                prob_also_f = (1.0 - af) * prob_blend_also + af * p_lr_a
            else:
                prob_also_f = prob_blend_also
        elif meta_cls == "lr":
            prob_also_f = clf.predict_proba(X_also_fit)[:, 1].astype(np.float64)
        elif meta_cls == "xgboost":
            prob_also_f = _meta_xgb_proba_positive_mean(xgb_meta_clfs, X_also_fit).astype(np.float64)
        else:
            raise SystemExit(f"--also-holdout-npz 不支持 meta_classifier={meta_cls!r}")
        test_metrics_also_holdout_npz = metrics_block(y_also, prob_also_f)
        prob_also_for_export = prob_also_f

    tm = metrics_block(yt, prob_te)
    vm = metrics_block(yv, prob_va)

    esd = getattr(args, "export_split_scores_dir", None)
    if esd is not None:
        outd = esd.resolve()
        _write_sid_y_p_csv(outd / "val.csv", sids_val, yv, np.asarray(prob_va, dtype=np.float64))
        _write_sid_y_p_csv(outd / "test.csv", sids_te, yt, np.asarray(prob_te, dtype=np.float64))
        if also_bundle is not None and prob_also_for_export is not None and y_also is not None:
            _write_sid_y_p_csv(
                outd / "also.csv", sids_also, y_also, np.asarray(prob_also_for_export, dtype=np.float64)
            )
        print("Wrote split scores under", outd)

    extrapolation_val_pick_report: Optional[Dict[str, object]] = None
    test_metrics_extrapolation_val_picked: Optional[Dict[str, float]] = None
    also_extrapolation_val_pick_report: Optional[Dict[str, object]] = None
    test_metrics_also_extrapolation_val_picked: Optional[Dict[str, float]] = None
    if args.holdout_npz is not None and getattr(args, "holdout_val_pick_triple_fusion", False):
        pev = val["p_esm"].astype(np.float64).reshape(-1)
        pgv = val["p_gearnet"].astype(np.float64).reshape(-1)
        if two_branch_npz:
            mean3_v = (pev + pgv) / 2.0
            median3_v = np.median(np.stack([pev, pgv], axis=0), axis=0)
        else:
            psv = val["p_saprot"].astype(np.float64).reshape(-1)
            mean3_v = (pev + psv + pgv) / 3.0
            median3_v = np.median(np.stack([pev, psv, pgv], axis=0), axis=0)
        vm_m3 = metrics_block(yv, mean3_v)
        vm_md = metrics_block(yv, median3_v)
        slack = float(getattr(args, "holdout_val_pick_mean3_vs_meta_val_slack", 0.0) or 0.0)
        mcc_m = float(vm["mcc"])
        mcc_3 = float(vm_m3["mcc"])
        mcc_md = float(vm_md["mcc"])
        tm_m3 = holdout_extra_reports["test_metrics_prob_mean3"]
        tm_md = holdout_extra_reports["test_metrics_prob_median3"]
        prio = {"median3": 3, "mean3": 2, "meta": 1}
        cands = [
            ("meta", mcc_m, tm, vm),
            ("mean3", mcc_3 + slack, tm_m3, vm_m3),
            ("median3", mcc_md + slack, tm_md, vm_md),
        ]
        head, _score, hold_pick, _val_block = max(cands, key=lambda x: (x[1], prio[x[0]]))
        extrapolation_val_pick_report = {
            "picked_head": head,
            "mode": "triple_val_mcc_tiebreak_median3_then_mean3_then_meta",
            "val_slack_applied_to_mean_and_median": slack,
            "rule": (
                "val 阈 0.5：比较 meta_MCC、mean3_MCC+slack、median3_MCC+slack，取最大者；"
                "平局优先 median3、mean3、Meta"
            ),
            "val_mcc_raw": {"meta": mcc_m, "mean3": mcc_3, "median3": mcc_md},
            "val_mcc_for_selection": {"meta": mcc_m, "mean3": mcc_3 + slack, "median3": mcc_md + slack},
            "val_metrics_meta_in_sample": vm,
            "val_metrics_mean3": vm_m3,
            "val_metrics_median3": vm_md,
            "holdout_metrics_meta": tm,
            "holdout_metrics_mean3": tm_m3,
            "holdout_metrics_median3": tm_md,
        }
        test_metrics_extrapolation_val_picked = dict(hold_pick)
    elif args.holdout_npz is not None and getattr(args, "holdout_val_pick_mean3_vs_meta", False):
        pev = val["p_esm"].astype(np.float64).reshape(-1)
        pgv = val["p_gearnet"].astype(np.float64).reshape(-1)
        if two_branch_npz:
            mean3_v = (pev + pgv) / 2.0
        else:
            psv = val["p_saprot"].astype(np.float64).reshape(-1)
            mean3_v = (pev + psv + pgv) / 3.0
        vm_m3 = metrics_block(yv, mean3_v)
        mcc_m = float(vm["mcc"])
        mcc_3 = float(vm_m3["mcc"])
        slack = float(getattr(args, "holdout_val_pick_mean3_vs_meta_val_slack", 0.0) or 0.0)
        pick = "mean3" if (mcc_3 + slack) > mcc_m else "meta"
        tm_m3 = holdout_extra_reports["test_metrics_prob_mean3"]
        extrapolation_val_pick_report = {
            "picked_head": pick,
            "val_slack_applied": slack,
            "rule": (
                "val 阈 0.5：若 (mean3_MCC + val_slack) > meta_MCC 则 holdout 推荐 mean3，否则 Meta；"
                "val_slack 默认 0 为严格比较"
            ),
            "val_metrics_meta_in_sample": vm,
            "val_metrics_mean3": vm_m3,
            "holdout_metrics_meta": tm,
            "holdout_metrics_mean3": tm_m3,
        }
        test_metrics_extrapolation_val_picked = dict(tm_m3 if pick == "mean3" else tm)

    if (
        bool(getattr(args, "also_val_pick_triple_fusion", False))
        and args.holdout_npz is None
        and also_bundle is not None
        and prob_also_f is not None
        and y_also is not None
        and test_metrics_also_holdout_npz is not None
    ):
        if "also_metrics_prob_mean3" not in holdout_extra_reports or "also_metrics_prob_median3" not in holdout_extra_reports:
            raise SystemExit(
                "内部错误：also triple 缺少 also_metrics_prob_mean3/median3（应已同开 --holdout-report-base-fusions）"
            )
        pev = val["p_esm"].astype(np.float64).reshape(-1)
        pgv = val["p_gearnet"].astype(np.float64).reshape(-1)
        if two_branch_npz:
            mean3_v = (pev + pgv) / 2.0
            median3_v = np.median(np.stack([pev, pgv], axis=0), axis=0)
        else:
            psv = val["p_saprot"].astype(np.float64).reshape(-1)
            mean3_v = (pev + psv + pgv) / 3.0
            median3_v = np.median(np.stack([pev, psv, pgv], axis=0), axis=0)
        vm_m3 = metrics_block(yv, mean3_v)
        vm_md = metrics_block(yv, median3_v)
        slack = float(getattr(args, "holdout_val_pick_mean3_vs_meta_val_slack", 0.0) or 0.0)
        mcc_m = float(vm["mcc"])
        mcc_3 = float(vm_m3["mcc"])
        mcc_md = float(vm_md["mcc"])
        am = dict(test_metrics_also_holdout_npz)
        am_m3 = holdout_extra_reports["also_metrics_prob_mean3"]
        am_md = holdout_extra_reports["also_metrics_prob_median3"]
        prio = {"median3": 3, "mean3": 2, "meta": 1}
        cands_also = [
            ("meta", mcc_m, am, vm),
            ("mean3", mcc_3 + slack, am_m3, vm_m3),
            ("median3", mcc_md + slack, am_md, vm_md),
        ]
        ahead, _ascore, also_pick, _aval = max(cands_also, key=lambda x: (x[1], prio[x[0]]))
        also_extrapolation_val_pick_report = {
            "picked_head": ahead,
            "mode": "triple_val_mcc_tiebreak_median3_then_mean3_then_meta_on_also_npz",
            "val_slack_applied_to_mean_and_median": slack,
            "rule": (
                "在 val 上比较 meta_MCC、mean3_MCC+slack、median3_MCC+slack（与 holdout triple 同规则）；"
                "将所选一侧在 --also-holdout-npz 上的指标写入 test_metrics_also_extrapolation_val_picked"
            ),
            "val_mcc_raw": {"meta": mcc_m, "mean3": mcc_3, "median3": mcc_md},
            "val_mcc_for_selection": {"meta": mcc_m, "mean3": mcc_3 + slack, "median3": mcc_md + slack},
            "val_metrics_meta_in_sample": vm,
            "val_metrics_mean3": vm_m3,
            "val_metrics_median3": vm_md,
            "also_metrics_meta": am,
            "also_metrics_mean3": am_m3,
            "also_metrics_median3": am_md,
        }
        test_metrics_also_extrapolation_val_picked = dict(also_pick)

    eval_score_profile_strata: Optional[Dict[str, object]] = None
    if args.export_scores_csv is not None:
        eval_score_profile_strata = export_eval_score_profile(
            csv_path=args.export_scores_csv.resolve(),
            sids=sids_te,
            yt=yt,
            prob_te=np.asarray(prob_te, dtype=np.float64),
            bundle=test,
            Xt_unscaled=Xt,
            miss_col=int(miss_col),
            seq_map=seq_map,
        )
        strata_path = args.export_scores_csv.with_name(args.export_scores_csv.stem + "_strata.json")
        strata_path.write_text(
            json.dumps(eval_score_profile_strata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Wrote", args.export_scores_csv, "and", strata_path)

    meta_lr_coef_report: Optional[Dict[str, object]] = None
    if meta_cls == "lr":
        if args.meta_moe_missing and hasattr(clf_rich, "coef_") and getattr(clf_rich, "coef_", None) is not None:
            wr = np.asarray(clf_rich.coef_, dtype=np.float64).reshape(-1)
            wp = np.asarray(clf_poor.coef_, dtype=np.float64).reshape(-1)
            meta_lr_coef_report = {
                "layout": "moe_two_logistic",
                "rich_expert": {
                    "intercept": float(np.asarray(clf_rich.intercept_).reshape(-1)[0]),
                    "coef_l2_total": float(np.linalg.norm(wr)),
                    "per_block": summarize_lr_coef_blocks(clf_rich.coef_, col_blocks),
                },
                "poor_expert": {
                    "intercept": float(np.asarray(clf_poor.intercept_).reshape(-1)[0]),
                    "coef_l2_total": float(np.linalg.norm(wp)),
                    "per_block": summarize_lr_coef_blocks(clf_poor.coef_, col_blocks),
                },
            }
        elif not args.meta_moe_missing and hasattr(clf, "coef_") and getattr(clf, "coef_", None) is not None:
            w = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
            meta_lr_coef_report = {
                "layout": "single_logistic",
                "intercept": float(np.asarray(clf.intercept_).reshape(-1)[0]),
                "coef_l2_total": float(np.linalg.norm(w)),
                "per_block": summarize_lr_coef_blocks(clf.coef_, col_blocks),
            }

    threshold_scan_report: Optional[Dict[str, object]] = None
    if getattr(args, "threshold_scan", False):
        obj = str(getattr(args, "threshold_scan_objective", "mcc") or "mcc").lower()
        if obj == "acc":
            thr_star, opt_star = scan_threshold_max_acc(yv, prob_va)
            obj_key = "maximize_accuracy_on_val_probs"
        else:
            thr_star, opt_star = scan_threshold_max_mcc(yv, prob_va)
            obj_key = "maximize_matthews_corrcoef_on_val_probs"
        pred_va_thr = (np.asarray(prob_va, dtype=np.float64) >= thr_star).astype(np.int64)
        mcc_at_thr = float(metrics.matthews_corrcoef(yv, pred_va_thr))
        acc_at_thr = float(np.mean(pred_va_thr == yv))
        threshold_scan_report = {
            "objective": obj_key,
            "threshold_scan_objective": obj,
            "grid_linear_0p01_to_0p99_199_steps": True,
            "best_threshold": thr_star,
            "val_optimized_score": opt_star,
            "val_mcc_at_best_threshold": mcc_at_thr,
            "val_accuracy_at_best_threshold": acc_at_thr,
            "on_platt_calibrated_probs": bool(getattr(args, "platt_calibrate", False)),
            "val": {
                "threshold_tuned": metrics_block(yv, prob_va, threshold=thr_star),
                "default_0p5": metrics_block(yv, prob_va),
            },
            "eval_test_or_holdout": {
                "threshold_tuned": metrics_block(yt, prob_te, threshold=thr_star),
                "default_0p5": metrics_block(yt, prob_te),
            },
        }

    fig_dim = 0
    if args.fig_extra:
        fig_dim = (8 if two_branch_npz else 12) + 3 + 1
    if args.meta_moe_missing:
        base_method = "stacking_meta_moe_missing_lr_plus_abc"
    elif meta_cls == "xgboost":
        base_method = "stacking_meta_xgb_plus_abc"
        if int(getattr(args, "meta_xgb_seed_ensemble_k", 1) or 1) > 1:
            base_method = f"{base_method}_seedens{int(args.meta_xgb_seed_ensemble_k)}"
        if int(getattr(args, "meta_xgb_bootstrap_k", 1) or 1) > 1:
            base_method = f"{base_method}_boot{int(args.meta_xgb_bootstrap_k)}"
    else:
        base_method = (
            "stacking_meta_lr_aux_xgb_val_blend_plus_abc"
            if aux_lr_xgb_blend
            else "stacking_meta_lr_plus_abc"
        )
        if aux_lr_xgb_blend and meta_lr_fuse_primary_after_aux_blend_report is not None:
            base_method = f"{base_method}_fuselrmeta"
    method_str = base_method + ("_fig" if args.fig_extra else "")
    coef_shape: List[int] = []
    if args.meta_moe_missing and hasattr(clf_rich, "coef_") and getattr(clf_rich, "coef_", None) is not None:
        coef_shape = [list(np.asarray(clf_rich.coef_).shape), list(np.asarray(clf_poor.coef_).shape)]
    elif hasattr(clf, "coef_") and getattr(clf, "coef_", None) is not None:
        coef_shape = list(np.asarray(clf.coef_).shape)
    spw_report = _resolve_xgb_scale_pos_weight(args, y_train_meta) if meta_cls == "xgboost" else None
    report = {
        "method": method_str,
        "meta_classifier": meta_cls,
        "meta_lr_use_rbf_sample_weight": (
            bool(getattr(args, "meta_lr_use_rbf_sample_weight", True))
            if meta_cls == "lr"
            else None
        ),
        "xgb_meta_hyperparams": (
            {
                "n_estimators": int(args.xgb_n_estimators),
                "max_depth": int(args.xgb_max_depth),
                "learning_rate": float(args.xgb_learning_rate),
                "reg_lambda": float(args.xgb_reg_lambda),
                "min_child_weight": float(args.xgb_min_child_weight),
                "subsample": float(args.xgb_subsample),
                "colsample_bytree": float(args.xgb_colsample_bytree),
                "scale_pos_weight": spw_report,
                "xgb_auto_scale_pos_weight": bool(getattr(args, "xgb_auto_scale_pos_weight", False)),
                "xgb_scale_pos_weight_cli": getattr(args, "xgb_scale_pos_weight", None),
            }
            if meta_cls == "xgboost"
            else None
        ),
        "meta_xgb_sample_weight_missing_alpha": (
            float(getattr(args, "meta_xgb_sample_weight_missing_alpha", 0.0) or 0.0)
            if meta_cls == "xgboost"
            else 0.0
        ),
        "meta_xgb_bootstrap_k": (
            int(getattr(args, "meta_xgb_bootstrap_k", 1) or 1) if meta_cls == "xgboost" else 1
        ),
        "meta_xgb_bootstrap_seed": (
            int(getattr(args, "meta_xgb_bootstrap_seed", 2026) or 2026) if meta_cls == "xgboost" else 2026
        ),
        "meta_xgb_seed_ensemble_k": (
            int(getattr(args, "meta_xgb_seed_ensemble_k", 1) or 1) if meta_cls == "xgboost" else 1
        ),
        "meta_xgb_seed_ensemble_random_states": (
            [
                int(getattr(args, "meta_xgb_bootstrap_seed", 2026) or 2026) + j
                for j in range(int(getattr(args, "meta_xgb_seed_ensemble_k", 1) or 1))
            ]
            if meta_cls == "xgboost" and int(getattr(args, "meta_xgb_seed_ensemble_k", 1) or 1) > 1
            else None
        ),
        "meta_val_reweight_holdout_rbf_gamma": float(
            getattr(args, "meta_val_reweight_holdout_rbf_gamma", 0.0) or 0.0
        ),
        "meta_val_reweight_holdout_alpha": float(
            getattr(args, "meta_val_reweight_holdout_alpha", 0.5) or 0.5
        ),
        "meta_val_reweight_holdout_max_rows": int(
            getattr(args, "meta_val_reweight_holdout_max_rows", 512) or 512
        ),
        "meta_val_reweight_holdout_rbf_avg_seeds": int(
            getattr(args, "meta_val_reweight_holdout_rbf_avg_seeds", 1) or 1
        ),
        "npz_two_branch_esm_gear": bool(two_branch_npz),
        "meta_cross_features": bool(args.meta_cross_features),
        "meta_moe_missing": bool(args.meta_moe_missing),
        "moe_gate_low": float(args.moe_gate_low),
        "moe_gate_high": float(args.moe_gate_high),
        "moe_expert_boost": float(args.moe_expert_boost),
        "missing_ratio_col": int(miss_col),
        "stacking_penalty": args.stacking_penalty,
        "stacking_C": float(args.stacking_C),
        "scale_meta": bool(args.scale_meta),
        "fig_extra": bool(args.fig_extra),
        "plddt_csv": str(args.plddt_csv) if args.plddt_csv else None,
        "plddt_val_hits": len(hit_vals),
        "plddt_fill_value": plddt_fill,
        "breakdown": {
            "base_probs": int(n_base_probs_meta),
            "prob_hadamard_pairs": int(hadamard_meta_dim),
            "kmer_pca": int(args.kmer_pca_dim),
            "missing_ratio": 1,
            "fig_disagreement_logit_ratio": int(fig_disagreement_cols),
            "phys_pi_gravy_thr": 3 if args.fig_extra else 0,
            "mean_plddt": 1 if args.fig_extra else 0,
            "fig_extra_total": fig_dim,
            "robust_ood_extra_dim": int(robust_dim),
        },
        "robust_ood_meta": bool(args.robust_ood_meta),
        "robust_plddt_center": float(args.robust_plddt_center),
        "robust_plddt_scale": float(args.robust_plddt_scale),
        "afp_local_motif": bool(args.afp_local_motif),
        "afp_local_motif_dim": int(motif_dim),
        "meta_drop_hadamard": bool(getattr(args, "meta_drop_hadamard", False)),
        "meta_seq_len": bool(args.meta_seq_len),
        "meta_seq_len_dim": int(seq_len_dim),
        "val_median_aa_len_for_meta_seq": float(val_median_aa_len_for_meta),
        "kmer_k": KMER_K,
        "kmer_raw_dim": KMER_DIM,
        "val_metrics_in_sample": vm,
        "test_metrics": tm,
        "test_metrics_also_holdout_npz": test_metrics_also_holdout_npz,
        "also_holdout_npz": str(also_npz_p.resolve()) if also_npz_p is not None else None,
        "also_holdout_note": (
            "与 test_metrics 同一 Meta fit 与 val 选 w；Xt 为 --also-holdout-npz；标签仅用于算指标、不参与选参"
            if test_metrics_also_holdout_npz is not None
            else None
        ),
        "train_oof_npz": str(train_oof_npz_p.resolve()) if train_oof_npz_p is not None else None,
        "train_oof_raw_rows_before_val_drop": int(train_oof_raw_n) if train_oof_npz_p is not None else 0,
        "train_oof_val_overlap_dropped": int(train_oof_val_overlap_dropped) if train_oof_npz_p is not None else 0,
        "n_train_oof_meta_rows_after_val_drop": int(X_oof.shape[0]) if X_oof is not None else 0,
        "meta_fit_used_train_oof_augment": bool(X_oof_fit is not None and y_oof is not None),
        "meta_train_oof_augment_row_weight": float(getattr(args, "meta_train_oof_augment_row_weight", 1.0) or 1.0),
        "export_split_scores_dir": str(esd.resolve()) if esd is not None else None,
        "holdout_npz": str(args.holdout_npz.resolve()) if args.holdout_npz else None,
        "holdout_note": (
            "test_metrics 对应 holdout_npz（非官方 test）" if args.holdout_npz else "test_metrics 对应官方 test_npz"
        ),
        "meta_val_rbf_reference_holdout_npz": str(ref_npz_p.resolve()) if ref_npz_p else None,
        "meta_val_rbf_reference_pos_fasta": (
            str(args.meta_val_rbf_reference_pos_fasta.resolve())
            if getattr(args, "meta_val_rbf_reference_pos_fasta", None)
            else None
        ),
        "meta_val_rbf_reference_neg_fasta": (
            str(args.meta_val_rbf_reference_neg_fasta.resolve())
            if getattr(args, "meta_val_rbf_reference_neg_fasta", None)
            else None
        ),
        "meta_val_rbf_reference_3di_fasta": (
            str(args.meta_val_rbf_reference_3di_fasta.resolve())
            if getattr(args, "meta_val_rbf_reference_3di_fasta", None)
            else None
        ),
        "coef_shape": coef_shape,
        "feature_dim": int(Xv.shape[1]),
        "note_focal_loss": "图4 Focal γ 微调需在底层分类训练脚本中改损失并重训，本脚本仅 Meta 层",
        "functional_csv_blocks": func_metas,
        "feat_auto_pca_dim": pca_fdim,
        "meta_lr_coef_report": meta_lr_coef_report,
        "threshold_scan_report": threshold_scan_report,
        "platt_calibrate": bool(getattr(args, "platt_calibrate", False)),
        "platt_calibration": platt_report,
        "metrics_on_platt_calibrated_probs": bool(getattr(args, "platt_calibrate", False)),
        "eval_score_profile_strata": eval_score_profile_strata,
        "meta_mean3_blend_report": mean3_blend_report,
        "meta_lr_aux_xgb_blend": bool(aux_lr_xgb_blend),
        "meta_lr_aux_xgb_blend_report": meta_lr_aux_xgb_blend_report,
        "meta_lr_fuse_primary_after_aux_blend_report": meta_lr_fuse_primary_after_aux_blend_report,
        "extrapolation_val_pick_report": extrapolation_val_pick_report,
        "test_metrics_extrapolation_val_picked": test_metrics_extrapolation_val_picked,
        "also_extrapolation_val_pick_report": also_extrapolation_val_pick_report,
        "test_metrics_also_extrapolation_val_picked": test_metrics_also_extrapolation_val_picked,
        **holdout_extra_reports,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    out_json = args.out_json or (
        ROOT / "trained" / "ensemble_prep_stacking_row19" / f"stacking_plus_abc_meta_lr{fig_suffix_default(args)}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print("Wrote", out_json)

    if args.append_csv:
        fig_desc = ""
        if args.fig_extra:
            fig_desc = "+分歧度/logit/理化锚点+pLDDT(PDB B-factor)"
        panel_note = "【四图一键 four_panel】" if args.four_panel else ""
        func_note = ""
        if func_metas:
            tags = [str(m.get("tag", "")) for m in func_metas]
            func_note = "+功能CSV（" + ",".join(tags) + "）"
        tag_prefix = (args.csv_row_tag_zh or "").strip()
        feat_title = (
            f"{panel_note}ESM2+GearNet（两路概率）+k-mer(PCA)+Hadamard+missing{fig_desc}{func_note}"
            if two_branch_npz
            else f"{panel_note}ESM2+SaProt+GearNet（三路概率）+k-mer(PCA)+Hadamard+missing{fig_desc}{func_note}"
        )
        if getattr(args, "robust_ood_meta", False):
            feat_title += "+稳健Meta(pLDDT门控×p+ppi行统计)"
        if getattr(args, "afp_local_motif", False):
            feat_title += f"+局部motif(Thr/TxT等{AFP_LOCAL_MOTIF_DIM}维)"
        if getattr(args, "meta_seq_len", False):
            feat_title += f"+长度门控(log1p(L),L/val中位,{META_SEQ_LEN_DIM}维交互)"
        if args.meta_cross_features:
            feat_title += (
                "+Meta交叉(p_esm·meanPPI,p_gearnet·GOpc1)"
                if two_branch_npz
                else "+Meta交叉(p_esm·meanPPI,p_saprot·GOpc1)"
            )
        if args.meta_moe_missing:
            feat_title += "+Meta-MoE(missing软门控双LR)"
        if getattr(args, "platt_calibrate", False):
            feat_title += "+Platt(val校准Meta分数)"
        if getattr(args, "meta_drop_hadamard", False):
            feat_title += "（无Hadamard概率交互3维）"
        if getattr(args, "meta_mean3_blend", False):
            feat_title += "+val选权·Meta与三路均值凸组合"
        if getattr(args, "meta_lr_aux_xgb_blend", False):
            feat_title += "+val选权·Meta-LR与辅XGB概率凸组合"
        if tag_prefix:
            feat_title = tag_prefix + feat_title
        if args.meta_moe_missing:
            clf_desc = (
                f"Meta-MoE/missing：双LR（{args.stacking_penalty},C={args.stacking_C}）"
                f"val 上 sample_weight 强调 missing<{args.moe_gate_low} 与 >{args.moe_gate_high}；"
                f"推理 p=(1-g)p_rich+g·poor，g=clip((miss-low)/(high-low))；功能PCA={pca_fdim}"
            )
        elif meta_cls == "xgboost":
            clf_desc = (
                f"Meta-XGBoost（hist，depth={args.xgb_max_depth}，n_est={args.xgb_n_estimators}，"
                f"η={args.xgb_learning_rate:g}，reg_lambda≈{args.xgb_reg_lambda:g}"
                f"{'，输入已标准化' if args.scale_meta else ''}；"
                f"功能PCA维={pca_fdim}；底层 Focal 需重训三路后再 ensemble）"
            )
            a_mw = float(getattr(args, "meta_xgb_sample_weight_missing_alpha", 0.0) or 0.0)
            if a_mw > 0.0:
                clf_desc += f"；val fit 按 missing_ratio 加权（alpha={a_mw:g}）"
            kb = int(getattr(args, "meta_xgb_bootstrap_k", 1) or 1)
            if kb > 1:
                clf_desc += f"；bootstrap K={kb} 概率平均（seed={int(getattr(args, 'meta_xgb_bootstrap_seed', 2026))}）"
            k_se = int(getattr(args, "meta_xgb_seed_ensemble_k", 1) or 1)
            if k_se > 1:
                s0 = int(getattr(args, "meta_xgb_bootstrap_seed", 2026) or 2026)
                clf_desc += f"；seed 集成 K={k_se}（random_state {s0}…{s0 + k_se - 1}，全 val 各训一棵，概率平均）"
            g_rbf = float(getattr(args, "meta_val_reweight_holdout_rbf_gamma", 0.0) or 0.0)
            if g_rbf > 0.0:
                clf_desc += (
                    f"；val 按与 holdout Meta 特征 RBF 相似度加权（gamma={g_rbf:g}，"
                    f"alpha={float(getattr(args, 'meta_val_reweight_holdout_alpha', 0.5) or 0.5):g}；无 holdout 标签）"
                )
        else:
            clf_desc = (
                f"Meta-LR（penalty={args.stacking_penalty}, C={args.stacking_C}"
                f"{'，标准化' if args.scale_meta else ''}；"
                f"底层 Focal 需 --loss focal 重训三路后再 ensemble）"
            )
        if args.robust_ood_meta:
            clf_desc += "；+稳健外推：pLDDT门控×三路p + PPI行L2/|·|/stub"
        if args.afp_local_motif:
            clf_desc += f"；+局部motif {AFP_LOCAL_MOTIF_DIM}维（Thr/TxT 统计，非CGR）"
        if args.meta_seq_len:
            clf_desc += f"；+长度门控 {META_SEQ_LEN_DIM}维（L/val中位×p_esm/p_gearnet）"
        if args.meta_cross_features:
            clf_desc += "；+两列交叉特征"
        if getattr(args, "platt_calibrate", False):
            clf_desc += "；+Platt：val 上 sigmoid(a·score+b) 校准正类概率（表内为校准后·0.5 阈）"
        if getattr(args, "meta_drop_hadamard", False):
            clf_desc += "；消融：去掉三路 p 的 Hadamard 交互（3 维）"
        if getattr(args, "meta_mean3_blend", False):
            clf_desc += (
                f"；+凸组合：p=(1-w)p_meta+w·mean(p)，w∈[0,{args.meta_mean3_blend_w_max}]"
                f" 在 val 上网格选优，val_MCC≥基线−{args.meta_mean3_blend_mcc_slack}"
            )
        if getattr(args, "meta_lr_aux_xgb_blend", False):
            clf_desc += (
                f"；+LR/辅XGB 概率凸组合：p=(1-w)p_LR+w·p_XGB，w∈[0,{args.meta_lr_aux_xgb_blend_w_max}]"
                f" 在 val 上选优：val_MCC≥仅LR−{args.meta_lr_aux_xgb_blend_mcc_slack}"
                f"{f'；目标 val_MCC−{args.meta_lr_aux_xgb_blend_val_penalty_lambda:g}·w' if float(getattr(args, 'meta_lr_aux_xgb_blend_val_penalty_lambda', 0.0) or 0.0) > 0 else ''}"
            )
        fusion_desc = (
            "val_npz 训练 Meta；holdout_npz 推理（独立集）；val 与主 B 线同源"
            if args.holdout_npz
            else "val_npz 训练 Meta；test_npz 推理；与最强三路概率 NPZ 同源"
        )
        if getattr(args, "platt_calibrate", False):
            fusion_desc += "；test 指标基于 val 拟合 Platt 后的概率"
        zh = {
            "特征": feat_title,
            "分类器": clf_desc,
            "融合器": fusion_desc,
            "结构分支": (
                "missing_ratio 来自 3Di；"
                "pLDDT 来自 CSV 或常数填充；"
                "理化：Biopython pI + KD 疏水 + Thr%"
            ),
            "test_accuracy": f"{tm['accuracy']:.4f}",
            "test_mcc": f"{tm['mcc']:.4f}",
            "test_auc": f"{tm['auc']:.4f}",
            "test_aupr": f"{tm['aupr']:.4f}",
            "test_sensitivity": f"{tm['sensitivity']:.4f}",
            "test_recall": f"{tm['recall']:.4f}",
            "test_specificity": f"{tm['specificity']:.4f}",
            "test_f1": f"{tm['f1']:.4f}",
        }
        append_experiments_table_row(args.experiments_csv.resolve(), zh)
        print("Appended row to", args.experiments_csv)


def fig_suffix_default(args: argparse.Namespace) -> str:
    parts = [f"_{args.stacking_penalty}"]
    if args.scale_meta:
        parts.append("_scaled")
    if getattr(args, "meta_moe_missing", False):
        parts.append("_moe")
    if not args.fig_extra:
        parts.append("_nofig")
    return "".join(parts)


if __name__ == "__main__":
    main()
