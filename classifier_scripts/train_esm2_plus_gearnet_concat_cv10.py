#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from binary_focal_loss import make_ce_or_focal

ROOT = Path(__file__).resolve().parents[1]


def _models_root() -> Path:
    import os
    return Path(os.environ.get("AFP_MODELS_ROOT", Path.home() / "models"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta_pairs(path: Path, label: int) -> List[Tuple[str, int, str]]:
    rows: List[Tuple[str, int, str]] = []
    sid = ""
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid:
                    seq = "".join(seq_parts).upper()
                    if not seq:
                        raise ValueError(f"{path}:{line_no} 上一条序列为空")
                    rows.append((sid, label, seq))
                sid = line[1:].strip().split()[0]
                seq_parts = []
            else:
                if not sid:
                    raise ValueError(f"{path}:{line_no} 序列出现在首个 header 前")
                seq_parts.append(line)
    if sid:
        seq = "".join(seq_parts).upper()
        if not seq:
            raise ValueError(f"{path}: 最后一条序列为空")
        rows.append((sid, label, seq))
    return rows


def load_split(pos_fasta: Path, neg_fasta: Path) -> List[Tuple[str, int, str]]:
    rows = read_fasta_pairs(pos_fasta, 1) + read_fasta_pairs(neg_fasta, 0)
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("检测到重复 seq_id")
    return rows


def build_concat_matrix(esm_feat: np.ndarray, gear_feat: np.ndarray, gear_mask: np.ndarray) -> np.ndarray:
    return np.concatenate([esm_feat, gear_feat, gear_mask.astype(np.float32)[:, None]], axis=1).astype(np.float32)


def mrmr_greedy_indices(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    mRMR 风格贪心：相关性用 mutual_info_classif(X_i, y)，冗余用与已选维度的平均 |线性相关|。
    预计算标准化后的 Gram 矩阵 G≈corr，复杂度约 O(nD^2 + kD)，避免逐对相关慢循环。
    """
    D = int(X.shape[1])
    k = min(int(k), D)
    mi = mutual_info_classif(X, y, random_state=seed)
    Xs = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s = Xs.std(axis=0)
    s[s < 1e-8] = 1.0
    Xn = (Xs - Xs.mean(axis=0)) / s
    n = float(max(Xn.shape[0], 1))
    G = (Xn.T @ Xn) / n
    np.fill_diagonal(G, 0.0)

    selected: List[int] = []
    first = int(np.argmax(mi))
    selected.append(first)
    candidates = set(range(D))
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_j = -1
        best_score = -1e100
        sel_arr = np.array(selected, dtype=np.int64)
        for j in candidates:
            rel = float(mi[j])
            red = float(np.mean(np.abs(G[j, sel_arr])))
            score = rel - red
            if score > best_score:
                best_score = score
                best_j = int(j)
        if best_j < 0:
            break
        selected.append(best_j)
        candidates.remove(best_j)

    return np.array(selected, dtype=np.int64)


def select_feature_indices(X: np.ndarray, y: np.ndarray, method: str, top_k: int, seed: int) -> np.ndarray:
    """仅在当前折的训练集上拟合选择器，返回列索引（长度≤top_k）。"""
    D = int(X.shape[1])
    k = min(int(top_k), D)
    m = (method or "none").strip().lower()
    if m in ("", "none"):
        return np.arange(D, dtype=np.int64)
    if m == "rf":
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt",
        )
        rf.fit(X, y)
        imp = rf.feature_importances_
        order = np.argsort(imp)[::-1][:k]
        return order.astype(np.int64)
    if m == "xgb":
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as e:
            raise RuntimeError("未安装 xgboost：pip install xgboost，或改用 --feature_select rf") from e
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        clf.fit(X, y)
        imp = clf.feature_importances_
        order = np.argsort(imp)[::-1][:k]
        return order.astype(np.int64)
    if m == "mrmr":
        return mrmr_greedy_indices(X, y, k, seed)
    raise ValueError(f"未知 --feature_select: {method}（支持 none/rf/xgb/mrmr）")


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = float(metrics.accuracy_score(y_true, y_pred))
    mcc = float(metrics.matthews_corrcoef(y_true, y_pred))
    try:
        auc = float(metrics.roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    p, r, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = float(metrics.auc(r, p))
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(metrics.f1_score(y_true, y_pred))
    return {"accuracy": acc, "mcc": mcc, "auc": auc, "aupr": aupr, "sensitivity": sens, "recall": sens, "specificity": spec, "f1": f1}


def _agg_numeric(folds: List[Dict[str, Any]], path: List[str]) -> Dict[str, float]:
    vals: List[float] = []
    for f in folds:
        x: Any = f
        for p in path:
            x = x[p]
        if isinstance(x, float) and np.isnan(x):
            continue
        vals.append(float(x))
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    a = np.array(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a)}


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


def load_esm2_from_cache(feature_dir: Path, all_ids: List[str]) -> Tuple[np.ndarray, int]:
    seq_path = feature_dir / "seq_ids.json"
    esm_path = feature_dir / "esm2_token_bins.npy"
    if not seq_path.is_file() or not esm_path.is_file():
        raise FileNotFoundError(f"特征目录缺少 seq_ids.json / esm2_token_bins.npy: {feature_dir}")

    with seq_path.open("r", encoding="utf-8") as f:
        cached_ids: List[str] = json.load(f)
    row_of = {sid: i for i, sid in enumerate(cached_ids)}
    missing = [sid for sid in all_ids if sid not in row_of]
    if missing:
        raise ValueError(f"ESM2 cache 缺少 {len(missing)} 个 seq_id（示例）: {missing[:5]}")

    esm_all = np.load(esm_path)  # [N,T,D]
    esm_mean = esm_all.mean(axis=1).astype(np.float32)  # [N,D]
    dim = int(esm_mean.shape[-1])
    out = np.empty((len(all_ids), dim), dtype=np.float32)
    for i, sid in enumerate(all_ids):
        out[i] = esm_mean[row_of[sid]]
    return out, dim


def load_torchdrug_gearnet_class():
    import torchdrug

    # 避免 torchdrug.models.__init__ 触发 ESM/torchvision 导入冲突
    gm_path = Path(torchdrug.__file__).resolve().parent / "models" / "gearnet.py"
    spec = importlib.util.spec_from_file_location("td_gearnet_module", gm_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.GeometryAwareRelationalGraphNeuralNetwork


def patch_sparse_coo_fallback() -> None:
    import torchdrug.layers.conv as td_conv
    import torchdrug.utils as td_utils

    # 绕开 torchdrug 的 C++ JIT 扩展（当前环境会在 crypt.h 上失败）
    def _safe_sparse(indices: torch.Tensor, values: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return torch.sparse_coo_tensor(indices, values, size)

    td_utils.sparse_coo_tensor = _safe_sparse
    td_conv.utils.sparse_coo_tensor = _safe_sparse


@torch.no_grad()
def build_or_load_gearnet_embeddings(
    all_ids: List[str],
    pdb_dir: Path,
    checkpoint: Path,
    cache_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_path = cache_dir / "seq_ids.json"
    feat_path = cache_dir / "gearnet_mc_graph_feature.npy"
    mask_path = cache_dir / "gearnet_structure_mask.npy"
    fail_path = cache_dir / "failed_ids.json"

    if seq_path.is_file() and feat_path.is_file() and mask_path.is_file():
        with seq_path.open("r", encoding="utf-8") as f:
            cached_ids: List[str] = json.load(f)
        row_of = {sid: i for i, sid in enumerate(cached_ids)}
        emb_all = np.load(feat_path).astype(np.float32)
        mask_all = np.load(mask_path).astype(np.float32)
        dim = int(emb_all.shape[-1])
        out = np.zeros((len(all_ids), dim), dtype=np.float32)
        out_mask = np.zeros((len(all_ids),), dtype=np.float32)
        missing = []
        for i, sid in enumerate(all_ids):
            j = row_of.get(sid, None)
            if j is None:
                missing.append(sid)
                continue
            out[i] = emb_all[j]
            out_mask[i] = mask_all[j]
        if missing:
            raise ValueError(f"GearNet embedding cache 缺少 {len(missing)} 个 seq_id（示例）: {missing[:5]}")
        return out, out_mask, dim

    from torchdrug import data, layers, transforms

    patch_sparse_coo_fallback()
    GearNet = load_torchdrug_gearnet_class()
    model = GearNet(
        input_dim=21,
        hidden_dims=[512, 512, 512, 512, 512, 512],
        batch_norm=True,
        concat_hidden=True,
        short_cut=True,
        readout="sum",
        num_relation=7,
        edge_input_dim=59,
        num_angle_bin=8,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    model.eval()

    graph_builder = layers.GraphConstruction(
        node_layers=[layers.geometry.AlphaCarbonNode()],
        edge_layers=[
            layers.geometry.SequentialEdge(max_distance=2),
            layers.geometry.SpatialEdge(radius=10.0, min_distance=5),
            layers.geometry.KNNEdge(k=10, min_distance=5),
        ],
        edge_feature="gearnet",
    )

    emb_list: List[np.ndarray] = []
    mask_list: List[float] = []
    failed_ids: List[str] = []

    for i, sid in enumerate(all_ids, start=1):
        pdb_path = pdb_dir / f"{sid}.pdb"
        if not pdb_path.is_file():
            emb_list.append(np.zeros((3072,), dtype=np.float32))
            mask_list.append(0.0)
            failed_ids.append(sid)
            continue
        try:
            p = data.Protein.from_pdb(str(pdb_path), atom_feature=None, bond_feature=None, residue_feature="default")
            pp = data.Protein.pack([p])
            pp = transforms.ProteinView(view="residue")({"graph": pp})["graph"]
            g = graph_builder(pp)
            g = g.to(device)
            out = model(g, g.node_feature.float())
            vec = out["graph_feature"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            emb_list.append(vec)
            mask_list.append(1.0)
        except Exception:
            emb_list.append(np.zeros((3072,), dtype=np.float32))
            mask_list.append(0.0)
            failed_ids.append(sid)

        if i % 500 == 0:
            print(f"[GearNet emb] {i}/{len(all_ids)} done, missing_or_failed={len(failed_ids)}")

    emb_all = np.stack(emb_list, axis=0).astype(np.float32)
    mask_all = np.asarray(mask_list, dtype=np.float32)
    with seq_path.open("w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)
    np.save(feat_path, emb_all)
    np.save(mask_path, mask_all)
    with fail_path.open("w", encoding="utf-8") as f:
        json.dump(failed_ids, f, ensure_ascii=False, indent=2)
    return emb_all, mask_all, int(emb_all.shape[-1])


class FeatureMatrixDataset(Dataset):
    """行由 full_X 索引，列可选 feat_idx 子集（与训练折上的特征选择一致）。"""

    def __init__(self, row_idx: np.ndarray, labels: np.ndarray, full_X: np.ndarray, feat_idx: Optional[np.ndarray] = None):
        self.row_idx = row_idx.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.full_X = full_X
        self.feat_idx = None if feat_idx is None else feat_idx.astype(np.int64)

    def __len__(self) -> int:
        return int(self.row_idx.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = int(self.row_idx[idx])
        x = self.full_X[i]
        if self.feat_idx is not None:
            x = x[self.feat_idx]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor(self.labels[idx], dtype=torch.long)


class GearNetSubsetConcatDataset(Dataset):
    """ESM2 全维 + GearNet 按列子集 + mask 标量拼接；用于仅在结构分支上选维。"""

    def __init__(
        self,
        row_idx: np.ndarray,
        labels: np.ndarray,
        esm_feat: np.ndarray,
        gear_feat: np.ndarray,
        gear_mask: np.ndarray,
        gear_col_idx: np.ndarray,
    ):
        self.row_idx = row_idx.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.esm_feat = esm_feat
        self.gear_feat = gear_feat
        self.gear_mask = gear_mask.astype(np.float32)
        self.gear_col_idx = gear_col_idx.astype(np.int64)

    def __len__(self) -> int:
        return int(self.row_idx.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = int(self.row_idx[idx])
        e = self.esm_feat[i]
        g = self.gear_feat[i, self.gear_col_idx]
        m = np.array([float(self.gear_mask[i])], dtype=np.float32)
        x = np.concatenate([e, g, m], axis=0)
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(self.labels[idx], dtype=torch.long)


class EsmGearMaskDataset(Dataset):
    """逐样本返回 ESM 向量、GearNet 向量、结构可用标量 mask（供加权双头使用）。"""

    def __init__(
        self,
        row_idx: np.ndarray,
        labels: np.ndarray,
        esm_feat: np.ndarray,
        gear_feat: np.ndarray,
        gear_mask: np.ndarray,
    ):
        self.row_idx = row_idx.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.esm_feat = esm_feat
        self.gear_feat = gear_feat
        self.gear_mask = gear_mask.astype(np.float32)

    def __len__(self) -> int:
        return int(self.row_idx.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(self.row_idx[idx])
        e = torch.from_numpy(self.esm_feat[i].astype(np.float32))
        g = torch.from_numpy(self.gear_feat[i].astype(np.float32))
        m = torch.tensor(float(self.gear_mask[i]), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return e, g, m, y


class WeightedDualHeadClassifier(nn.Module):
    """
    序列 / 结构各一套 MLP→2 类 logits；可学习权重 w=softmax(θ)，有结构时 logits=w0*le+w1*lg，
    无结构（mask=0）时退化为仅序列头 le，避免零向量结构拖累。
    """

    def __init__(self, esm_dim: int, gear_dim: int, hidden_dims: Tuple[int, ...], dropout: float):
        super().__init__()

        def _mlp_head(in_dim: int) -> nn.Sequential:
            layers_: List[nn.Module] = []
            d = in_dim
            for h in hidden_dims:
                layers_.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
                d = h
            layers_.append(nn.Linear(d, 2))
            return nn.Sequential(*layers_)

        self.head_esm = _mlp_head(esm_dim)
        self.head_gear = _mlp_head(gear_dim)
        self.branch_weight = nn.Parameter(torch.zeros(2))

    def forward(self, esm: torch.Tensor, gear: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        le = self.head_esm(esm)
        lg = self.head_gear(gear)
        w = F.softmax(self.branch_weight, dim=0)
        if mask.dim() == 1:
            m = mask.unsqueeze(-1)
        else:
            m = mask
        m = m.clamp(0.0, 1.0).to(dtype=le.dtype)
        blended = w[0] * le + w[1] * lg
        logits = m * blended + (1.0 - m) * le
        return logits


class ResidualGateDualHeadClassifier(nn.Module):
    """
    残差式融合：logits = logits_esm + λ * logits_gear，λ = mask * σ(Gate(ESM, Gear, mask))。
    无结构（mask=0）时 λ=0，退化为纯 ESM 头；有结构时 Gate 学习对 Gear 头 logits 的缩放修正。
    """

    def __init__(
        self,
        esm_dim: int,
        gear_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
        gate_hidden: int = 128,
    ):
        super().__init__()

        def _mlp_head(in_dim: int) -> nn.Sequential:
            layers_: List[nn.Module] = []
            d = in_dim
            for h in hidden_dims:
                layers_.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
                d = h
            layers_.append(nn.Linear(d, 2))
            return nn.Sequential(*layers_)

        self.head_esm = _mlp_head(esm_dim)
        self.head_gear = _mlp_head(gear_dim)
        gate_in_dim = esm_dim + gear_dim + 1
        gh = max(8, int(gate_hidden))
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gh),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gh, 1),
        )

    def forward(self, esm: torch.Tensor, gear: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        le = self.head_esm(esm)
        lg = self.head_gear(gear)
        if mask.dim() == 1:
            m = mask.unsqueeze(-1)
        else:
            m = mask
        m = m.clamp(0.0, 1.0).to(dtype=esm.dtype)
        gate_in = torch.cat([esm, gear, m], dim=-1)
        lam = torch.sigmoid(self.gate(gate_in))
        lam = lam * m
        return le + lam * lg


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], dropout: float):
        super().__init__()
        layers_: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers_.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers_.append(nn.Linear(d, 2))
        self.cls = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x)

    def first_linear(self) -> nn.Linear:
        m0 = self.cls[0]
        if not isinstance(m0, nn.Linear):
            raise RuntimeError("MLPClassifier 首层应为 Linear")
        return m0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    train: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    l1_first_lambda: float = 0.0,
) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train and l1_first_lambda > 0.0 and isinstance(model, MLPClassifier):
                loss = loss + l1_first_lambda * model.first_linear().weight.abs().sum()
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())

    m = _metrics_block(np.array(all_true), np.array(all_pred), np.array(all_prob))
    m["loss"] = total_loss / max(n_batches, 1)
    return m


def run_epoch_dual_head(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    train: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []

    with torch.set_grad_enabled(train):
        for e, g, mask, y in loader:
            e = e.to(device)
            g = g.to(device)
            mask = mask.to(device)
            y = y.to(device)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            logits = model(e, g, mask)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())

    m = _metrics_block(np.array(all_true), np.array(all_pred), np.array(all_prob))
    m["loss"] = total_loss / max(n_batches, 1)
    return m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESM2 + GearNet (pretrained) concat classifier on AFP-MetaStack")
    p.add_argument("--train_pos_fasta", type=Path, default=ROOT / "data" / "train_set" / "pos_train.fasta")
    p.add_argument("--train_neg_fasta", type=Path, default=ROOT / "data" / "train_set" / "neg_train.fasta")
    p.add_argument("--test_pos_fasta", type=Path, default=ROOT / "data" / "test_set" / "pos_test.fasta")
    p.add_argument("--test_neg_fasta", type=Path, default=ROOT / "data" / "test_set" / "neg_test.fasta")
    p.add_argument("--esm2_cache_dir", type=Path, default=ROOT / "features" / "afp_e_esm2_saprot_token_bins")
    p.add_argument("--gearnet_checkpoint", type=Path, default=_models_root() / "gearnet/checkpoints/mc_gearnet_edge.pth")
    p.add_argument("--pdb_dir", type=Path, default=ROOT / "data" / "raw" / "pdb")
    p.add_argument("--gearnet_emb_cache_dir", type=Path, default=ROOT / "features" / "afp_e_gearnet_mc_emb_cache")
    p.add_argument("--hidden_dims", type=str, default="512,256")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="验证集 loss 连续若干 epoch 无改善则停止该 fold；0 表示关闭。",
    )
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument(
        "--lr_step_epochs",
        type=int,
        default=0,
        help="StepLR：每隔若干 epoch 将 lr 乘以 lr_gamma；0 表示不用。",
    )
    p.add_argument("--lr_gamma", type=float, default=0.95)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=Path, default=ROOT / "trained" / "esm2_plus_gearnet_mc_concat_cv10")
    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument("--experiments_csv", type=Path, default=ROOT / "results" / "experiments_table_zh.csv")
    p.add_argument(
        "--feature_select",
        type=str,
        default="none",
        choices=["none", "rf", "xgb", "mrmr"],
        help="MLP 前的选维：none=全维；rf/xgb=树模型特征重要性 Top-K；mrmr=MI−相关冗余的贪心 mRMR 风格（无需额外 pip 包）",
    )
    p.add_argument("--fs_top_k", type=int, default=512, help="fs_scope=concat 时：全拼接向量上选维保留的维数上界")
    p.add_argument(
        "--fs_scope",
        type=str,
        default="concat",
        choices=["concat", "gearnet_only"],
        help="concat=在 ESM⊕GearNet⊕mask 全拼接上选维；gearnet_only=ESM2 与 mask 始终全保留，仅对 GearNet 列做选维",
    )
    p.add_argument(
        "--gear_fs_top_k",
        type=int,
        default=128,
        help="fs_scope=gearnet_only 且 feature_select≠none 时：从 GearNet 原始维数中保留的维数（如 64/128；与 gear_dim 取 min）",
    )
    p.add_argument(
        "--l1_first_lambda",
        type=float,
        default=0.0,
        help="首层 Linear 权重的 L1 惩罚（Lasso 式稀疏），与 CE 相加；0 表示关闭",
    )
    p.add_argument(
        "--classifier",
        type=str,
        default="concat_mlp",
        choices=["concat_mlp", "weighted_heads", "residual_gate"],
        help="concat_mlp=拼接 MLP；weighted_heads=双头+softmax(w) 混合；residual_gate=logit_esm+λ·logit_gear，λ=mask·σ(Gate)",
    )
    p.add_argument(
        "--gate_hidden",
        type=int,
        default=128,
        help="residual_gate 时 Gate MLP 隐层维（仅 classifier=residual_gate 使用）",
    )
    p.add_argument("--loss", type=str, default="ce", choices=("ce", "focal"), help="分类损失：ce 或 focal（图4 需重训分支）")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss γ，常用 1.5–2.5")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    if not hidden_dims:
        raise ValueError("hidden_dims 至少包含一层")

    if args.classifier in ("weighted_heads", "residual_gate"):
        if args.feature_select != "none":
            raise ValueError(f"{args.classifier} 与 --feature_select 暂不兼容，请使用 --feature_select none")
        if args.fs_scope != "concat":
            raise ValueError(f"{args.classifier} 请使用 --fs_scope concat（全维双头）")
        if args.l1_first_lambda != 0.0:
            raise ValueError(f"{args.classifier} 不支持 --l1_first_lambda，请设为 0")

    train_rows = load_split(args.train_pos_fasta.resolve(), args.train_neg_fasta.resolve())
    test_rows = load_split(args.test_pos_fasta.resolve(), args.test_neg_fasta.resolve())
    train_ids = [x[0] for x in train_rows]
    train_y = np.array([x[1] for x in train_rows], dtype=np.int64)
    test_ids = [x[0] for x in test_rows]
    test_y = np.array([x[1] for x in test_rows], dtype=np.int64)
    all_ids = list(dict.fromkeys(train_ids + test_ids))
    id2idx = {sid: i for i, sid in enumerate(all_ids)}
    train_idx = np.array([id2idx[s] for s in train_ids], dtype=np.int64)
    test_idx = np.array([id2idx[s] for s in test_ids], dtype=np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("train_n:", len(train_ids), "test_n:", len(test_ids))

    set_seed(args.seed)
    esm_feat, esm_dim = load_esm2_from_cache(args.esm2_cache_dir.resolve(), all_ids)
    print("esm2_cache_dir:", args.esm2_cache_dir.resolve(), "esm_dim:", esm_dim)

    gear_feat, gear_mask, gear_dim = build_or_load_gearnet_embeddings(
        all_ids=all_ids,
        pdb_dir=args.pdb_dir.resolve(),
        checkpoint=args.gearnet_checkpoint.resolve(),
        cache_dir=args.gearnet_emb_cache_dir.resolve(),
        device=device,
    )
    print("gearnet_checkpoint:", args.gearnet_checkpoint.resolve())
    print("gearnet_emb_cache_dir:", args.gearnet_emb_cache_dir.resolve(), "gear_dim:", gear_dim)
    print("structure_available(train/test):", f"{gear_mask[train_idx].mean():.4f}/{gear_mask[test_idx].mean():.4f}")
    print("classifier:", args.classifier)

    full_X = build_concat_matrix(esm_feat, gear_feat, gear_mask)
    concat_dim = int(full_X.shape[1])
    print("concat_dim (ESM⊕GearNet⊕mask):", concat_dim)
    print("fs_scope:", args.fs_scope)
    if args.feature_select != "none":
        if args.fs_scope == "gearnet_only":
            print("feature_select (仅 GearNet 分支):", args.feature_select, "gear_fs_top_k:", args.gear_fs_top_k)
        else:
            print("feature_select (全拼接):", args.feature_select, "fs_top_k:", args.fs_top_k)
    if args.l1_first_lambda > 0.0:
        print("l1_first_lambda:", args.l1_first_lambda)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds: List[Dict[str, Any]] = []
    for fold, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(train_y)), train_y)):
        set_seed(args.seed + fold)
        tr_idx = train_idx[tr_i]
        va_idx = train_idx[va_i]
        tr_y = train_y[tr_i]
        va_y = train_y[va_i]

        gear_idx_arr: Optional[np.ndarray] = None
        feat_idx_arr: Optional[np.ndarray] = None
        in_dim: int

        if args.classifier in ("weighted_heads", "residual_gate"):
            in_dim = concat_dim
            tr_ds = EsmGearMaskDataset(tr_idx, tr_y, esm_feat, gear_feat, gear_mask)
            va_ds = EsmGearMaskDataset(va_idx, va_y, esm_feat, gear_feat, gear_mask)
            te_ds = EsmGearMaskDataset(test_idx, test_y, esm_feat, gear_feat, gear_mask)
        elif args.fs_scope == "concat":
            X_tr_fit = full_X[tr_idx]
            feat_idx_arr = select_feature_indices(X_tr_fit, tr_y, args.feature_select, args.fs_top_k, args.seed + fold)
            in_dim = int(feat_idx_arr.shape[0])
            tr_ds = FeatureMatrixDataset(tr_idx, tr_y, full_X, feat_idx_arr)
            va_ds = FeatureMatrixDataset(va_idx, va_y, full_X, feat_idx_arr)
            te_ds = FeatureMatrixDataset(test_idx, test_y, full_X, feat_idx_arr)
        else:
            if (args.feature_select or "none").strip().lower() in ("", "none"):
                gear_idx_arr = np.arange(gear_dim, dtype=np.int64)
            else:
                Xg_tr = gear_feat[tr_idx]
                gear_idx_arr = select_feature_indices(
                    Xg_tr, tr_y, args.feature_select, args.gear_fs_top_k, args.seed + fold
                )
            in_dim = esm_dim + int(gear_idx_arr.shape[0]) + 1
            tr_ds = GearNetSubsetConcatDataset(tr_idx, tr_y, esm_feat, gear_feat, gear_mask, gear_idx_arr)
            va_ds = GearNetSubsetConcatDataset(va_idx, va_y, esm_feat, gear_feat, gear_mask, gear_idx_arr)
            te_ds = GearNetSubsetConcatDataset(test_idx, test_y, esm_feat, gear_feat, gear_mask, gear_idx_arr)

        pin = device.type == "cuda"
        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
        va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

        if args.classifier == "weighted_heads":
            model = WeightedDualHeadClassifier(esm_dim, gear_dim, hidden_dims, args.dropout).to(device)
        elif args.classifier == "residual_gate":
            model = ResidualGateDualHeadClassifier(
                esm_dim, gear_dim, hidden_dims, args.dropout, gate_hidden=args.gate_hidden
            ).to(device)
        else:
            model = MLPClassifier(in_dim, hidden_dims, args.dropout).to(device)
        criterion = make_ce_or_focal(args.loss, args.focal_gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if int(args.lr_step_epochs) > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=int(args.lr_step_epochs), gamma=float(args.lr_gamma)
            )

        fold_out = args.out_dir / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_no_improve = 0
        stopped_epoch = int(args.epochs)

        if args.classifier == "weighted_heads":
            print(f"\n=== Fold {fold + 1}/{args.n_splits} (加权双头: ESM+GearNet, mask 控制融合) ===")
        elif args.classifier == "residual_gate":
            print(f"\n=== Fold {fold + 1}/{args.n_splits} (残差门控: logits_esm + λ·logits_gear, λ=mask·σ(Gate)) ===")
        elif args.fs_scope == "concat":
            print(f"\n=== Fold {fold + 1}/{args.n_splits} (ESM2 + GearNet concat) in_dim={in_dim}/{concat_dim} ===")
        else:
            assert gear_idx_arr is not None
            print(
                f"\n=== Fold {fold + 1}/{args.n_splits} (仅 GearNet 选维) ESM={esm_dim} "
                f"Gear_sel={int(gear_idx_arr.shape[0])}/{gear_dim} in_dim={in_dim} ==="
            )
        for ep in range(args.epochs):
            if args.classifier in ("weighted_heads", "residual_gate"):
                tr_m = run_epoch_dual_head(model, tr_ld, device, criterion, train=True, optimizer=optimizer)
                va_m = run_epoch_dual_head(model, va_ld, device, criterion, train=False)
            else:
                tr_m = run_epoch(
                    model,
                    tr_ld,
                    device,
                    criterion,
                    train=True,
                    optimizer=optimizer,
                    l1_first_lambda=args.l1_first_lambda,
                )
                va_m = run_epoch(model, va_ld, device, criterion, train=False, l1_first_lambda=0.0)
            if scheduler is not None:
                scheduler.step()
            print(
                f"  Epoch {ep + 1}/{args.epochs} train_loss={tr_m['loss']:.4f} "
                f"val_loss={va_m['loss']:.4f} val_acc={va_m['accuracy']:.4f}"
            )
            improved = float(va_m["loss"]) < (best_val - float(args.early_stop_min_delta))
            if improved:
                best_val = float(va_m["loss"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                if args.save_checkpoints:
                    torch.save(
                        {"model_state_dict": best_state, "epoch": ep, "val_loss": best_val},
                        fold_out / "best_model.pt",
                    )
            else:
                epochs_no_improve += 1

            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                stopped_epoch = ep + 1
                print(f"  Early stopping at epoch {stopped_epoch} (patience={args.early_stop_patience})")
                break

        if best_state is None:
            raise RuntimeError("no best state")
        model.load_state_dict(best_state)
        fs: Dict[str, Any] = {
            "fold": fold,
            "epochs_ran": int(stopped_epoch),
            "effective_train_n": len(tr_ds),
            "effective_val_n": len(va_ds),
            "effective_test_n": len(te_ds),
        }
        fs["classifier"] = args.classifier
        fs["fs_scope"] = args.fs_scope
        fs["feature_select"] = args.feature_select
        fs["fs_top_k"] = int(args.fs_top_k)
        fs["gear_fs_top_k"] = int(args.gear_fs_top_k)
        fs["concat_dim"] = concat_dim
        fs["selected_dim"] = in_dim
        if args.classifier == "weighted_heads":
            w = F.softmax(model.branch_weight.detach(), dim=0).cpu().numpy().tolist()
            fs["branch_softmax_weights_esm_gear"] = w
        elif args.classifier == "residual_gate":
            fs["residual_gate_formula"] = "logits = logits_esm + (mask * sigmoid(Gate(esm,gear,mask))) * logits_gear"
            fs["gate_hidden"] = int(args.gate_hidden)
        elif args.fs_scope == "concat":
            assert feat_idx_arr is not None
            fs["selected_idx_preview"] = feat_idx_arr[: min(64, feat_idx_arr.shape[0])].astype(int).tolist()
        else:
            assert gear_idx_arr is not None
            fs["gear_dim"] = gear_dim
            fs["gear_selected_dim"] = int(gear_idx_arr.shape[0])
            fs["gear_idx_preview"] = gear_idx_arr[: min(64, gear_idx_arr.shape[0])].astype(int).tolist()
        if args.classifier in ("weighted_heads", "residual_gate"):
            fs["val"] = run_epoch_dual_head(model, va_ld, device, criterion, train=False)
        else:
            fs["val"] = run_epoch(model, va_ld, device, criterion, train=False, l1_first_lambda=0.0)
        fs["val"].pop("loss", None)
        if args.classifier in ("weighted_heads", "residual_gate"):
            fs["test"] = run_epoch_dual_head(model, te_ld, device, criterion, train=False)
        else:
            fs["test"] = run_epoch(model, te_ld, device, criterion, train=False, l1_first_lambda=0.0)
        fs["test"].pop("loss", None)
        folds.append(fs)

    if args.classifier == "weighted_heads":
        method_key = "ESM2_plus_GearNet_weighted_dual_head"
    elif args.classifier == "residual_gate":
        method_key = "ESM2_plus_GearNet_residual_gate_dual_head"
    else:
        method_key = "ESM2_plus_GearNet_pretrained_concat"
    report: Dict[str, Any] = {
        "method": method_key,
        "dataset": "AFP-MetaStack",
        "classifier": args.classifier,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "lr_step_epochs": args.lr_step_epochs,
        "lr_gamma": args.lr_gamma,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed_base": args.seed,
        "concat_dim": concat_dim,
        "fs_scope": args.fs_scope,
        "feature_select": args.feature_select,
        "fs_top_k": int(args.fs_top_k),
        "gear_fs_top_k": int(args.gear_fs_top_k),
        "l1_first_lambda": float(args.l1_first_lambda),
        "gate_hidden": int(args.gate_hidden) if args.classifier == "residual_gate" else None,
        "esm2_cache_dir": str(args.esm2_cache_dir.resolve()),
        "gearnet_checkpoint": str(args.gearnet_checkpoint.resolve()),
        "gearnet_emb_cache_dir": str(args.gearnet_emb_cache_dir.resolve()),
        "pdb_dir": str(args.pdb_dir.resolve()),
        "gear_dim": gear_dim,
        "structure_available_ratio_train": float(gear_mask[train_idx].mean()),
        "structure_available_ratio_test": float(gear_mask[test_idx].mean()),
        "folds": folds,
        "aggregate_val": {
            "val_accuracy": _agg_numeric(folds, ["val", "accuracy"]),
            "val_mcc": _agg_numeric(folds, ["val", "mcc"]),
            "val_auc": _agg_numeric(folds, ["val", "auc"]),
            "val_aupr": _agg_numeric(folds, ["val", "aupr"]),
            "val_sensitivity": _agg_numeric(folds, ["val", "sensitivity"]),
            "val_recall": _agg_numeric(folds, ["val", "recall"]),
            "val_specificity": _agg_numeric(folds, ["val", "specificity"]),
            "val_f1": _agg_numeric(folds, ["val", "f1"]),
        },
        "aggregate_test": {
            "test_accuracy": _agg_numeric(folds, ["test", "accuracy"]),
            "test_mcc": _agg_numeric(folds, ["test", "mcc"]),
            "test_auc": _agg_numeric(folds, ["test", "auc"]),
            "test_aupr": _agg_numeric(folds, ["test", "aupr"]),
            "test_sensitivity": _agg_numeric(folds, ["test", "sensitivity"]),
            "test_recall": _agg_numeric(folds, ["test", "recall"]),
            "test_specificity": _agg_numeric(folds, ["test", "specificity"]),
            "test_f1": _agg_numeric(folds, ["test", "f1"]),
        },
    }
    if args.classifier == "weighted_heads":
        out_json = args.out_dir / f"cv_metrics_esm2_plus_gearnet_weighted_heads_on_afp_e_cv{args.n_splits}.json"
    elif args.classifier == "residual_gate":
        out_json = args.out_dir / f"cv_metrics_esm2_plus_gearnet_residual_gate_on_afp_e_cv{args.n_splits}.json"
    else:
        out_json = args.out_dir / f"cv_metrics_esm2_plus_gearnet_mc_concat_on_afp_e_cv{args.n_splits}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"Wrote {out_json}")

    at = report["aggregate_test"]
    dims_s = "→".join(str(h) for h in hidden_dims) + "→2"
    fs_note = ""
    if args.fs_scope == "gearnet_only":
        if args.feature_select != "none":
            fs_note += f"；仅GearNet选维={args.feature_select}（Top-{args.gear_fs_top_k}），ESM2全{esm_dim}维"
        else:
            fs_note += f"；gearnet_only+none=GearNet全{gear_dim}维"
    elif args.feature_select != "none":
        fs_note += f"；全拼接选维={args.feature_select}（Top-{args.fs_top_k}）"
    if args.l1_first_lambda > 0.0:
        fs_note += f"；首层L1 λ={args.l1_first_lambda:g}"
    if args.classifier == "weighted_heads":
        fus = "双头 logits：有结构时 blend=w0·ESM+w1·Gear（w=softmax 可学习）；无结构时=仅 ESM 头"
        cls_title = f"加权双头MLP（每路{dims_s}；θ→softmax 融合）"
    elif args.classifier == "residual_gate":
        fus = f"logits=logits_ESM+λ·logits_Gear，λ=mask·σ(Gate)，Gate隐层={args.gate_hidden}"
        cls_title = f"残差门控双头MLP（每路{dims_s}）"
    elif args.fs_scope == "gearnet_only":
        fus = (
            f"ESM2全{esm_dim}维⊕GearNet分支"
            + (f"RF/XGB/mRMR选Top-{args.gear_fs_top_k}" if args.feature_select != "none" else f"全{gear_dim}维")
            + "⊕mask→MLP"
        )
        cls_title = f"Concat-MLP（{dims_s}）{fs_note}".strip()
    else:
        fus = f"简单拼接（ESM2⊕GearNet⊕mask）→{'按折选维' if args.feature_select != 'none' else '全维'}→MLP"
        cls_title = f"Concat-MLP（{dims_s}）{fs_note}".strip()
    zh_row = {
        "特征": "ESM2-t33-650M(mean-pool) + GearNet-MC预训练图特征",
        "分类器": cls_title,
        "融合器": fus,
        "结构分支": "PDB几何图（缺结构样本以零向量回退）",
        "test_accuracy": f"{at['test_accuracy']['mean']:.4f}",
        "test_mcc": f"{at['test_mcc']['mean']:.4f}",
        "test_auc": f"{at['test_auc']['mean']:.4f}",
        "test_aupr": f"{at['test_aupr']['mean']:.4f}",
        "test_sensitivity": f"{at['test_sensitivity']['mean']:.4f}",
        "test_recall": f"{at['test_recall']['mean']:.4f}",
        "test_specificity": f"{at['test_specificity']['mean']:.4f}",
        "test_f1": f"{at['test_f1']['mean']:.4f}",
    }
    zh_fields = list(zh_row.keys())
    zh_path = args.out_dir / "experiment_row_zh.csv"
    with zh_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=zh_fields)
        w.writeheader()
        w.writerow(zh_row)
    print(f"Wrote {zh_path}")

    append_experiments_table_row(args.experiments_csv.resolve(), zh_row)
    print(f"Appended row to {args.experiments_csv.resolve()}")


if __name__ == "__main__":
    main()

