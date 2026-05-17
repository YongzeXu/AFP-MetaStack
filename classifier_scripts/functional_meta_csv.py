"""
功能特征 CSV → 与 ensemble NPZ 按 seq_id 对齐，缺失列用验证集列中位数填充；
可选对宽表做 PCA（仅在 val 上 fit）。

CSV 约定：首列为 seq_id（或 id），其余列为数值特征（GO/EC 嵌入、PPI 向量、PSSM 统计、DeepLoc 概率、结合位点密度等）。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


def load_numeric_feature_csv(path: Path) -> Tuple[Dict[str, np.ndarray], List[str], int]:
    """返回 (sid -> float32 向量, 特征列名列表, 维数)。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: 空表头")
        cols = list(reader.fieldnames)
        id_candidates = {"seq_id", "id", "SID", "ID", "accession"}
        id_col = None
        for c in cols:
            if c.lower() in {x.lower() for x in id_candidates} or c == "seq_id":
                id_col = c
                break
        if id_col is None:
            id_col = cols[0]
        feat_names = [c for c in cols if c != id_col]
        dim = len(feat_names)
        if dim == 0:
            raise ValueError(f"{path}: 无数值列")
        out: Dict[str, np.ndarray] = {}
        for row in reader:
            sid = str(row.get(id_col, "")).strip()
            if not sid:
                continue
            vals: List[float] = []
            ok = True
            for c in feat_names:
                v = row.get(c, "")
                if v is None or str(v).strip() == "":
                    ok = False
                    break
                try:
                    vals.append(float(v))
                except ValueError:
                    ok = False
                    break
            if ok:
                out[sid] = np.array(vals, dtype=np.float32)
    return out, feat_names, dim


def _impute_and_optional_pca(
    X_val: np.ndarray,
    X_te: np.ndarray,
    *,
    auto_pca_dim: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[PCA]]:
    """缺失已在调用前处理；此处若启用 PCA 则压缩宽表。"""
    n_val, d = X_val.shape
    if auto_pca_dim <= 0 or d <= auto_pca_dim:
        return X_val.astype(np.float32), X_te.astype(np.float32), d, None
    k = min(auto_pca_dim, d, max(1, n_val))
    pca = PCA(n_components=k, random_state=random_state)
    Xv = pca.fit_transform(X_val).astype(np.float32)
    Xt = pca.transform(X_te).astype(np.float32)
    return Xv, Xt, k, pca


def align_csv_to_splits(
    sids_val: List[str],
    sids_te: List[str],
    path: Optional[Path],
    *,
    auto_pca_dim: int = 0,
    random_state: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    将单个 CSV 对齐到 val/test 行顺序；未见 seq_id 先用 **验证集矩阵列中位数** 填充，
    test 中未见 id 同样用 val 上得到的中位数（避免用 test 统计）。
    若 auto_pca_dim>0 且特征维数大于 auto_pca_dim，则在 val 上 PCA 到最多 auto_pca_dim 维。
    """
    meta: Dict[str, object] = {"path": str(path) if path else None, "dim_in": 0, "dim_out": 0, "pca": False}
    if path is None or not path.is_file():
        z = np.zeros((len(sids_val), 0), dtype=np.float32)
        zt = np.zeros((len(sids_te), 0), dtype=np.float32)
        return z, zt, meta

    tbl, names, dim = load_numeric_feature_csv(path.resolve())
    meta["columns"] = names[:12] + (["..."] if len(names) > 12 else [])
    meta["dim_in"] = dim

    M_val = np.full((len(sids_val), dim), np.nan, dtype=np.float64)
    for i, sid in enumerate(sids_val):
        if sid in tbl:
            M_val[i] = tbl[sid].astype(np.float64)
    med = np.nanmedian(M_val, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    ind = np.where(np.isnan(M_val))
    if ind[0].size > 0:
        M_val[ind] = np.take(med, ind[1])

    X_te = np.full((len(sids_te), dim), np.nan, dtype=np.float64)
    for i, sid in enumerate(sids_te):
        if sid in tbl:
            X_te[i] = tbl[sid].astype(np.float64)
        else:
            X_te[i] = med
    ind_te = np.where(np.isnan(X_te))
    if ind_te[0].size > 0:
        X_te[ind_te] = np.take(med, ind_te[1])

    Xv, Xt, dout, pca_obj = _impute_and_optional_pca(
        M_val, X_te, auto_pca_dim=auto_pca_dim, random_state=random_state
    )
    meta["dim_out"] = dout
    meta["pca"] = pca_obj is not None
    return Xv, Xt, meta


def align_csv_val_test_oof(
    sids_val: List[str],
    sids_te: List[str],
    sids_oof: List[str],
    path: Optional[Path],
    *,
    auto_pca_dim: int = 0,
    random_state: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """与 align_csv_to_splits 相同的中位数与 val 上 PCA，额外输出 train-OOF 行的变换后矩阵。"""
    meta: Dict[str, object] = {"path": str(path) if path else None, "dim_in": 0, "dim_out": 0, "pca": False, "oof_block": True}
    if path is None or not path.is_file():
        z = np.zeros((len(sids_val), 0), dtype=np.float32)
        zt = np.zeros((len(sids_te), 0), dtype=np.float32)
        zo = np.zeros((len(sids_oof), 0), dtype=np.float32)
        return z, zt, zo, meta

    tbl, names, dim = load_numeric_feature_csv(path.resolve())
    meta["columns"] = names[:12] + (["..."] if len(names) > 12 else [])
    meta["dim_in"] = dim

    M_val = np.full((len(sids_val), dim), np.nan, dtype=np.float64)
    for i, sid in enumerate(sids_val):
        if sid in tbl:
            M_val[i] = tbl[sid].astype(np.float64)
    med = np.nanmedian(M_val, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    ind = np.where(np.isnan(M_val))
    if ind[0].size > 0:
        M_val[ind] = np.take(med, ind[1])

    X_te = np.full((len(sids_te), dim), np.nan, dtype=np.float64)
    for i, sid in enumerate(sids_te):
        if sid in tbl:
            X_te[i] = tbl[sid].astype(np.float64)
        else:
            X_te[i] = med
    ind_te = np.where(np.isnan(X_te))
    if ind_te[0].size > 0:
        X_te[ind_te] = np.take(med, ind_te[1])

    M_oof = np.full((len(sids_oof), dim), np.nan, dtype=np.float64)
    for i, sid in enumerate(sids_oof):
        if sid in tbl:
            M_oof[i] = tbl[sid].astype(np.float64)
        else:
            M_oof[i] = med
    ind_oof = np.where(np.isnan(M_oof))
    if ind_oof[0].size > 0:
        M_oof[ind_oof] = np.take(med, ind_oof[1])

    Xv, Xt, dout, pca_obj = _impute_and_optional_pca(
        M_val, X_te, auto_pca_dim=auto_pca_dim, random_state=random_state
    )
    meta["dim_out"] = dout
    meta["pca"] = pca_obj is not None
    if pca_obj is not None:
        Xo = pca_obj.transform(M_oof).astype(np.float32)
    else:
        Xo = M_oof.astype(np.float32)
    return Xv, Xt, Xo, meta
