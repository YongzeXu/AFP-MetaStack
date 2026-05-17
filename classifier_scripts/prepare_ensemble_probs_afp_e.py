#!/usr/bin/env python3
"""
为 ensemble_stacking_plus_abc.py（或同类 stacking）准备对齐的 NPZ：

  - p_esm：仅 ESM2 mean-pool + MLP 的正类概率
  - p_saprot：ESM mean ⊕ SaProt mean 拼接 + MLP
  - p_gearnet：结构支路正类概率；``--gearnet_branch residual``=残差门控双头；
    ``weighted_heads``=可学习 softmax 双头融合（与全量 train 脚本一致）；
    ``multiview``=Contact+Lap+MTL（需 struct_aux_cache）。

两阶段：
  val：从 train 中分层划出 val_ratio 作为「调集成权重专用」验证集，三路在同一划分上训练并导出概率。
  test：对官方 test 集做 K 折（默认与 CV10 一致）：每折仅在 train 部分训练，预测 test，
       对 test 上概率做平均（与常见「CV 测试汇报」一致）。

依赖：与 GearNet 缓存脚本相同环境（torch）；GearNet 特征优先读 npy 缓存，无需重新跑图网络。

示例：
  conda activate afp-metastack
  cd /data/yongze/AFP-MetaStack && python3 classifier_scripts/prepare_ensemble_probs_afp_e.py --out_dir .../ensemble_probs_prep --phase both

  # 与主 CV 训练一致使用 Focal（需与 train_esm2_plus_gearnet_concat_cv10 的 --loss focal --focal_gamma 对齐）：
  python3 classifier_scripts/prepare_ensemble_probs_afp_e.py --loss focal --focal_gamma 1.5 ...

  # 仅导出训练集 OOF 三路概率（与 test 阶段相同的折内 early-stopping，预测在当折 holdout 行上；耗时≈ train_oof_splits 倍单折）：
  python3 classifier_scripts/prepare_ensemble_probs_afp_e.py --phase train_oof --train_oof_splits 5 --out_dir .../ensemble_probs_prep

  # 硬消融：可省略一路或两路一阶头（逗号分隔）；例：仅训 SaProt 头、esm 与 gearnet 写常数：
  python3 classifier_scripts/prepare_ensemble_probs_afp_e.py --omit_train_head esm,gearnet --omit_train_fill 0.5 --loss focal --focal_gamma 1.5 ...

  # 结构支路改为 Contact/Lap 多视角（仍写入 p_gearnet，供 Stacking 替换一路）：
  # 前置：python3 features_scripts/experimental/build_struct_multiview_cache.py
  python3 classifier_scripts/prepare_ensemble_probs_afp_e.py --gearnet_branch multiview \\
      --struct_aux_cache_dir .../afp_e_struct_multiview --out_dir .../ensemble_probs_prep_multiview --phase both \\
      --tta_runs 3 --tta_noise_std 0.01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))

import train_esm2_plus_gearnet_concat_cv10 as ge
import train_esm2_gearnet_multiview_mtl_cv10 as mvm
from binary_focal_loss import make_ce_or_focal

ROOT = Path(__file__).resolve().parents[1]


def _models_root() -> Path:
    return Path(os.environ.get("AFP_MODELS_ROOT", Path.home() / "models"))


class ArrayXYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)


def load_token_mean(feature_dir: Path, all_ids: List[str], fname: str) -> Tuple[np.ndarray, int]:
    seq_path = feature_dir / "seq_ids.json"
    arr_path = feature_dir / fname
    if not seq_path.is_file() or not arr_path.is_file():
        raise FileNotFoundError(f"缺少 {seq_path} 或 {arr_path}")
    with seq_path.open("r", encoding="utf-8") as f:
        cached_ids: List[str] = json.load(f)
    row_of = {sid: i for i, sid in enumerate(cached_ids)}
    missing = [sid for sid in all_ids if sid not in row_of]
    if missing:
        raise ValueError(f"{fname} cache 缺少 seq_id（示例）: {missing[:5]}")
    raw = np.load(arr_path)
    if raw.ndim == 3:
        feat = raw.mean(axis=1).astype(np.float32)
    else:
        feat = raw.astype(np.float32)
    dim = int(feat.shape[-1])
    out = np.empty((len(all_ids), dim), dtype=np.float32)
    for i, sid in enumerate(all_ids):
        out[i] = feat[row_of[sid]]
    return out, dim


def train_until_best_val(
    model: torch.nn.Module,
    tr_ld: DataLoader,
    va_ld: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
    dual_head: bool,
    crit: torch.nn.Module,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    no_va = va_ld is None or len(va_ld.dataset) == 0
    if no_va:
        for _ in range(epochs):
            if dual_head:
                ge.run_epoch_dual_head(model, tr_ld, device, crit, train=True, optimizer=opt)
            else:
                ge.run_epoch(model, tr_ld, device, crit, train=True, optimizer=opt, l1_first_lambda=0.0)
        return
    best = float("inf")
    best_sd = None
    for _ in range(epochs):
        if dual_head:
            ge.run_epoch_dual_head(model, tr_ld, device, crit, train=True, optimizer=opt)
            va_m = ge.run_epoch_dual_head(model, va_ld, device, crit, train=False)
        else:
            ge.run_epoch(model, tr_ld, device, crit, train=True, optimizer=opt, l1_first_lambda=0.0)
            va_m = ge.run_epoch(model, va_ld, device, crit, train=False, l1_first_lambda=0.0)
        if float(va_m["loss"]) < best:
            best = float(va_m["loss"])
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict(best_sd)


def split_fold_train_indices(
    tr_rel: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    inner_val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """折内训练索引（相对 train 数组下标）→ inner_tr, inner_va；若样本过少则 inner_va 为空。"""
    tr_rel = np.asarray(tr_rel).reshape(-1)
    if len(tr_rel) < 10:
        return tr_rel, np.array([], dtype=np.int64)
    try:
        tr_inner, va_inner = train_test_split(
            tr_rel,
            test_size=inner_val_ratio,
            stratify=train_y[tr_rel],
            random_state=seed,
        )
    except ValueError:
        tr_inner, va_inner = train_test_split(
            tr_rel,
            test_size=inner_val_ratio,
            random_state=seed,
        )
    return np.asarray(tr_inner).reshape(-1), np.asarray(va_inner).reshape(-1)


@torch.no_grad()
def predict_mlp_probs(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        b = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
        logits = model(b)
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        out.append(p)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_mlp_probs_tta(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    tta_runs: int,
    noise_std: float,
) -> np.ndarray:
    if tta_runs <= 1 or noise_std <= 0.0:
        return predict_mlp_probs(model, X, device, batch_size)
    model.eval()
    avg = np.zeros((X.shape[0],), dtype=np.float64)
    for _ in range(int(tta_runs)):
        noise = np.random.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        out: List[np.ndarray] = []
        for i in range(0, X.shape[0], batch_size):
            b = torch.from_numpy((X[i : i + batch_size] + noise[i : i + batch_size]).astype(np.float32)).to(device)
            logits = model(b)
            p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            out.append(p)
        avg += np.concatenate(out, axis=0).astype(np.float64)
    return (avg / float(tta_runs)).astype(np.float32)


@torch.no_grad()
def predict_residual_probs(
    model: ge.ResidualGateDualHeadClassifier,
    esm: np.ndarray,
    gear: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    n = esm.shape[0]
    for i in range(0, n, batch_size):
        e = torch.from_numpy(esm[i : i + batch_size].astype(np.float32)).to(device)
        g = torch.from_numpy(gear[i : i + batch_size].astype(np.float32)).to(device)
        m = torch.from_numpy(mask[i : i + batch_size].astype(np.float32)).to(device)
        logits = model(e, g, m)
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        out.append(p)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_residual_probs_tta(
    model: ge.ResidualGateDualHeadClassifier,
    esm: np.ndarray,
    gear: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
    tta_runs: int,
    noise_std: float,
) -> np.ndarray:
    if tta_runs <= 1 or noise_std <= 0.0:
        return predict_residual_probs(model, esm, gear, mask, device, batch_size)
    model.eval()
    avg = np.zeros((esm.shape[0],), dtype=np.float64)
    for _ in range(int(tta_runs)):
        ne = np.random.normal(0.0, noise_std, size=esm.shape).astype(np.float32)
        ng = np.random.normal(0.0, noise_std, size=gear.shape).astype(np.float32)
        out: List[np.ndarray] = []
        for i in range(0, esm.shape[0], batch_size):
            e = torch.from_numpy((esm[i : i + batch_size] + ne[i : i + batch_size]).astype(np.float32)).to(device)
            g = torch.from_numpy((gear[i : i + batch_size] + ng[i : i + batch_size]).astype(np.float32)).to(device)
            m = torch.from_numpy(mask[i : i + batch_size].astype(np.float32)).to(device)
            logits = model(e, g, m)
            p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            out.append(p)
        avg += np.concatenate(out, axis=0).astype(np.float64)
    return (avg / float(tta_runs)).astype(np.float32)


def train_until_best_val_multiview(
    model: torch.nn.Module,
    tr_ld: DataLoader,
    va_ld: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    wd: float,
    lambda_sasa: float,
    lambda_ctr: float,
    crit: torch.nn.Module,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    no_va = va_ld is None or len(va_ld.dataset) == 0
    if no_va:
        for _ in range(epochs):
            mvm.run_epoch_multiview(
                model,
                tr_ld,
                device,
                crit,
                lambda_sasa,
                lambda_ctr,
                train=True,
                optimizer=opt,
                entropy_infer=False,
                metrics_fn=ge._metrics_block,
            )
        return
    best = float("inf")
    best_sd = None
    for _ in range(epochs):
        mvm.run_epoch_multiview(
            model,
            tr_ld,
            device,
            crit,
            lambda_sasa,
            lambda_ctr,
            train=True,
            optimizer=opt,
            entropy_infer=False,
            metrics_fn=ge._metrics_block,
        )
        va_m = mvm.run_epoch_multiview(
            model,
            va_ld,
            device,
            crit,
            lambda_sasa,
            lambda_ctr,
            train=False,
            entropy_infer=False,
            metrics_fn=ge._metrics_block,
        )
        if float(va_m["loss"]) < best:
            best = float(va_m["loss"])
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict(best_sd)


@torch.no_grad()
def predict_multiview_probs(
    model: mvm.ResidualGateMultiViewMTL,
    esm: np.ndarray,
    gear: np.ndarray,
    mask: np.ndarray,
    contact: np.ndarray,
    lap: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    n = esm.shape[0]
    for i in range(0, n, batch_size):
        e = torch.from_numpy(esm[i : i + batch_size].astype(np.float32)).to(device)
        g = torch.from_numpy(gear[i : i + batch_size].astype(np.float32)).to(device)
        m = torch.from_numpy(mask[i : i + batch_size].astype(np.float32)).to(device)
        c = torch.from_numpy(contact[i : i + batch_size].astype(np.float32)).to(device).unsqueeze(1)
        lb = torch.from_numpy(lap[i : i + batch_size].astype(np.float32)).to(device)
        logits, _, _ = model(e, g, m, c, lb, entropy_blend_infer=False)
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        out.append(p)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_multiview_probs_tta(
    model: mvm.ResidualGateMultiViewMTL,
    esm: np.ndarray,
    gear: np.ndarray,
    mask: np.ndarray,
    contact: np.ndarray,
    lap: np.ndarray,
    device: torch.device,
    batch_size: int,
    tta_runs: int,
    noise_std: float,
) -> np.ndarray:
    if tta_runs <= 1 or noise_std <= 0.0:
        return predict_multiview_probs(model, esm, gear, mask, contact, lap, device, batch_size)
    model.eval()
    avg = np.zeros((esm.shape[0],), dtype=np.float64)
    for _ in range(int(tta_runs)):
        ne = np.random.normal(0.0, noise_std, size=esm.shape).astype(np.float32)
        ng = np.random.normal(0.0, noise_std, size=gear.shape).astype(np.float32)
        nc = np.random.normal(0.0, noise_std, size=contact.shape).astype(np.float32)
        nl = np.random.normal(0.0, noise_std, size=lap.shape).astype(np.float32)
        out: List[np.ndarray] = []
        for i in range(0, esm.shape[0], batch_size):
            e = torch.from_numpy((esm[i : i + batch_size] + ne[i : i + batch_size]).astype(np.float32)).to(device)
            g = torch.from_numpy((gear[i : i + batch_size] + ng[i : i + batch_size]).astype(np.float32)).to(device)
            m = torch.from_numpy(mask[i : i + batch_size].astype(np.float32)).to(device)
            c = torch.from_numpy((contact[i : i + batch_size] + nc[i : i + batch_size]).astype(np.float32)).to(device).unsqueeze(1)
            lb = torch.from_numpy((lap[i : i + batch_size] + nl[i : i + batch_size]).astype(np.float32)).to(device)
            logits, _, _ = model(e, g, m, c, lb, entropy_blend_infer=False)
            p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            out.append(p)
        avg += np.concatenate(out, axis=0).astype(np.float64)
    return (avg / float(tta_runs)).astype(np.float32)


def parse_omit_train_heads_arg(raw: str | None) -> frozenset[str]:
    """解析 --omit_train_head：单路 esm，或逗号多路 esm,gearnet（不可三者全省）。"""
    if raw is None or not str(raw).strip():
        return frozenset()
    parts = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    allowed = {"esm", "saprot", "gearnet"}
    for p in parts:
        if p not in allowed:
            raise SystemExit(f"--omit_train_head 含非法项 {p!r}，仅允许 esm,saprot,gearnet 及逗号组合")
    if len(parts) != len(set(parts)):
        raise SystemExit("--omit_train_head 含重复项")
    s = frozenset(parts)
    if len(s) == 3:
        raise SystemExit("不可同时省略全部三支路")
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出三路模型概率供 ensemble_stacking_plus_abc.py 等 Meta/stacking 使用")
    p.add_argument("--train_pos_fasta", type=Path, default=ROOT / "data" / "train_set" / "pos_train.fasta")
    p.add_argument("--train_neg_fasta", type=Path, default=ROOT / "data" / "train_set" / "neg_train.fasta")
    p.add_argument("--test_pos_fasta", type=Path, default=ROOT / "data" / "test_set" / "pos_test.fasta")
    p.add_argument("--test_neg_fasta", type=Path, default=ROOT / "data" / "test_set" / "neg_test.fasta")
    p.add_argument("--esm2_cache_dir", type=Path, default=ROOT / "features" / "afp_e_esm2_saprot_token_bins")
    p.add_argument("--gearnet_emb_cache_dir", type=Path, default=ROOT / "features" / "afp_e_gearnet_mc_emb_cache")
    p.add_argument("--gearnet_checkpoint", type=Path, default=_models_root() / "gearnet/checkpoints/mc_gearnet_edge.pth")
    p.add_argument("--pdb_dir", type=Path, default=ROOT / "data" / "raw" / "pdb")
    p.add_argument("--out_dir", type=Path, default=ROOT / "trained" / "ensemble_probs_prep")
    p.add_argument(
        "--phase",
        type=str,
        choices=["val", "test", "both", "train_oof"],
        default="both",
        help="val/test/both=原逻辑；train_oof=仅在完整 train 上做分层 K 折 OOF 概率（写出 train_oof_probs_for_ensemble.npz），不跑 val/test",
    )
    p.add_argument("--val_ratio", type=float, default=0.15, help="从 train 中划验证集比例（仅 phase 含 val）")
    p.add_argument("--n_splits", type=int, default=10, help="test 阶段 CV 折数，与主实验一致建议 10")
    p.add_argument(
        "--train_oof_splits",
        type=int,
        default=5,
        help="train_oof 阶段：训练集分层 K 折数（越大越准但越慢；可与 n_splits 不同）",
    )
    p.add_argument(
        "--inner_val_ratio",
        type=float,
        default=0.1,
        help="每折内从训练折再划验证集用于 early stopping；过小折可自动退回无验证",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dims", type=str, default="512,256")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument(
        "--gearnet_branch",
        type=str,
        choices=["residual", "multiview", "weighted_heads"],
        default="residual",
        help="residual=残差门控双头；weighted_heads=双头 softmax 融合（mask 控制）；multiview=Contact+Lap+MTL（需 struct_aux_cache）",
    )
    p.add_argument(
        "--struct_aux_cache_dir",
        type=Path,
        default=ROOT / "features" / "afp_e_struct_multiview",
        help="gearnet_branch=multiview 时：features_scripts/experimental/build_struct_multiview_cache.py 输出目录",
    )
    p.add_argument("--lap_dim", type=int, default=16)
    p.add_argument("--lambda_sasa", type=float, default=0.05)
    p.add_argument("--lambda_ctr", type=float, default=0.02)
    p.add_argument("--entropy_center", type=float, default=0.35)
    p.add_argument("--entropy_beta", type=float, default=8.0)
    p.add_argument("--tta_runs", type=int, default=1, help="测试增强采样次数；1 表示关闭")
    p.add_argument("--tta_noise_std", type=float, default=0.01, help="TTA 特征高斯扰动标准差")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--loss", type=str, default="ce", choices=("ce", "focal"), help="与 train_esm2_plus_gearnet_concat_cv10 一致：ce 或 focal")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="focal 时 γ，与主训练 --focal_gamma 对齐")
    p.add_argument(
        "--omit_train_head",
        type=str,
        default=None,
        help="硬消融：不训练哪些一阶头，逗号分隔（如 esm 或 esm,gearnet）；对应 p_* 整列写为 --omit_train_fill。另几路照常训。不可三者全省。",
    )
    p.add_argument(
        "--omit_train_fill",
        type=float,
        default=0.5,
        help="与 --omit_train_head 同时使用时，被省略支路的概率占位值。",
    )
    p.add_argument(
        "--npz_two_branch_esm_gear",
        action="store_true",
        help="主实验两路：不训练 SaProt 头，且 **NPZ 不写 p_saprot 键**（仅 p_esm、p_gearnet、seq_ids、y）。须与 ensemble_stacking_plus_abc 自动识别一致。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    omit_heads = parse_omit_train_heads_arg(args.omit_train_head)
    two_npz = bool(getattr(args, "npz_two_branch_esm_gear", False))
    if two_npz:
        if omit_heads and omit_heads != frozenset({"saprot"}):
            raise SystemExit("--npz_two_branch_esm_gear 与当前 --omit_train_head 冲突（应省略 saprot 或不写 omit_train_head）")
        omit_heads = frozenset({"saprot"})
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = ge.load_split(args.train_pos_fasta.resolve(), args.train_neg_fasta.resolve())
    test_rows = ge.load_split(args.test_pos_fasta.resolve(), args.test_neg_fasta.resolve())
    train_ids = [x[0] for x in train_rows]
    train_y = np.array([x[1] for x in train_rows], dtype=np.int64)
    test_ids = [x[0] for x in test_rows]
    test_y = np.array([x[1] for x in test_rows], dtype=np.int64)
    all_ids = list(dict.fromkeys(train_ids + test_ids))
    id2idx = {sid: i for i, sid in enumerate(all_ids)}
    train_idx = np.array([id2idx[s] for s in train_ids], dtype=np.int64)
    test_idx = np.array([id2idx[s] for s in test_ids], dtype=np.int64)

    ge.set_seed(args.seed)
    esm_feat, esm_dim = ge.load_esm2_from_cache(args.esm2_cache_dir.resolve(), all_ids)
    sap_feat, sap_dim = load_token_mean(args.esm2_cache_dir.resolve(), all_ids, "saprot_token_bins.npy")
    gear_feat, gear_mask, gear_dim = ge.build_or_load_gearnet_embeddings(
        all_ids=all_ids,
        pdb_dir=args.pdb_dir.resolve(),
        checkpoint=args.gearnet_checkpoint.resolve(),
        cache_dir=args.gearnet_emb_cache_dir.resolve(),
        device=device,
    )
    if args.gearnet_branch == "multiview":
        contact_all, lap_all, sasa_raw = mvm.align_multiview_arrays(
            all_ids, args.struct_aux_cache_dir.resolve(), args.lap_dim
        )
    else:
        contact_all = lap_all = sasa_raw = None  # type: ignore
    concat_esm_sap = np.concatenate([esm_feat, sap_feat], axis=1).astype(np.float32)
    concat_dim = int(concat_esm_sap.shape[1])

    pin = device.type == "cuda"
    crit = make_ce_or_focal(args.loss, args.focal_gamma, reduction="mean")
    meta = {
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "omit_train_heads": sorted(omit_heads) if omit_heads else None,
        "omit_train_fill": float(args.omit_train_fill) if omit_heads else None,
        "npz_two_branch_esm_gear": bool(two_npz),
        "esm_dim": esm_dim,
        "sap_dim": sap_dim,
        "gear_dim": gear_dim,
        "concat_esm_sap_dim": concat_dim,
        "device": str(device),
        "gearnet_branch": args.gearnet_branch,
        "loss": str(args.loss),
        "focal_gamma": float(args.focal_gamma),
    }
    if args.gearnet_branch == "multiview":
        meta["struct_aux_cache_dir"] = str(args.struct_aux_cache_dir.resolve())
        meta["lap_dim"] = int(args.lap_dim)
        meta["lambda_sasa"] = float(args.lambda_sasa)
        meta["lambda_ctr"] = float(args.lambda_ctr)

    if args.phase in ("val", "both"):
        ge.set_seed(args.seed)
        tr_sub_idx, va_idx_idx = train_test_split(
            np.arange(len(train_y)),
            test_size=args.val_ratio,
            stratify=train_y,
            random_state=args.seed,
        )
        tr_idx_abs = train_idx[tr_sub_idx]
        va_idx_abs = train_idx[va_idx_idx]
        y_tr = train_y[tr_sub_idx]
        y_va = train_y[va_idx_idx]

        fill_f = float(args.omit_train_fill)

        # --- ESM only ---
        if "esm" in omit_heads:
            p_esm_va = np.full(len(va_idx_abs), fill_f, dtype=np.float32)
        else:
            m_esm = ge.MLPClassifier(esm_dim, hidden_dims, args.dropout).to(device)
            ds_tr = ArrayXYDataset(esm_feat[tr_idx_abs], y_tr)
            ds_va = ArrayXYDataset(esm_feat[va_idx_abs], y_va)
            tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
            va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
            train_until_best_val(m_esm, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=False, crit=crit)
            p_esm_va = predict_mlp_probs(m_esm, esm_feat[va_idx_abs], device, args.batch_size)

        # --- ESM + SaProt concat ---
        if "saprot" in omit_heads:
            p_sap_va = np.full(len(va_idx_abs), fill_f, dtype=np.float32)
        else:
            m_sap = ge.MLPClassifier(concat_dim, hidden_dims, args.dropout).to(device)
            ds_tr = ArrayXYDataset(concat_esm_sap[tr_idx_abs], y_tr)
            ds_va = ArrayXYDataset(concat_esm_sap[va_idx_abs], y_va)
            tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
            va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
            train_until_best_val(m_sap, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=False, crit=crit)
            p_sap_va = predict_mlp_probs(m_sap, concat_esm_sap[va_idx_abs], device, args.batch_size)

        # --- 结构支路：残差门控 / 双头 softmax 融合 / Contact+Lap 多视角 ---
        if "gearnet" in omit_heads:
            p_ge_va = np.full(len(va_idx_abs), fill_f, dtype=np.float32)
        elif args.gearnet_branch == "residual":
            m_r = ge.ResidualGateDualHeadClassifier(
                esm_dim, gear_dim, hidden_dims, args.dropout, gate_hidden=args.gate_hidden
            ).to(device)
            ds_tr = ge.EsmGearMaskDataset(tr_idx_abs, y_tr, esm_feat, gear_feat, gear_mask)
            ds_va = ge.EsmGearMaskDataset(va_idx_abs, y_va, esm_feat, gear_feat, gear_mask)
            tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
            va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
            train_until_best_val(m_r, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=True, crit=crit)
            p_ge_va = predict_residual_probs(m_r, esm_feat[va_idx_abs], gear_feat[va_idx_abs], gear_mask[va_idx_abs], device, args.batch_size)
        elif args.gearnet_branch == "weighted_heads":
            m_w = ge.WeightedDualHeadClassifier(esm_dim, gear_dim, hidden_dims, args.dropout).to(device)
            ds_tr = ge.EsmGearMaskDataset(tr_idx_abs, y_tr, esm_feat, gear_feat, gear_mask)
            ds_va = ge.EsmGearMaskDataset(va_idx_abs, y_va, esm_feat, gear_feat, gear_mask)
            tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
            va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
            train_until_best_val(m_w, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=True, crit=crit)
            p_ge_va = predict_residual_probs(
                m_w, esm_feat[va_idx_abs], gear_feat[va_idx_abs], gear_mask[va_idx_abs], device, args.batch_size
            )
        else:
            assert sasa_raw is not None and contact_all is not None and lap_all is not None
            sm = float(np.mean(sasa_raw[tr_idx_abs]))
            ss = float(np.std(sasa_raw[tr_idx_abs]) + 1e-6)
            sasa_z = (sasa_raw - sm) / ss
            m_r = mvm.ResidualGateMultiViewMTL(
                esm_dim,
                gear_dim,
                args.lap_dim,
                hidden_dims,
                args.dropout,
                gate_hidden=args.gate_hidden,
                entropy_center=args.entropy_center,
                entropy_beta=args.entropy_beta,
            ).to(device)
            ds_tr = mvm.MultiViewDataset(tr_idx_abs, y_tr, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
            ds_va = mvm.MultiViewDataset(va_idx_abs, y_va, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
            tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
            va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
            train_until_best_val_multiview(
                m_r,
                tr_ld,
                va_ld,
                device,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.lambda_sasa,
                args.lambda_ctr,
                crit,
            )
            p_ge_va = predict_multiview_probs(
                m_r,
                esm_feat[va_idx_abs],
                gear_feat[va_idx_abs],
                gear_mask[va_idx_abs],
                contact_all[va_idx_abs],
                lap_all[va_idx_abs],
                device,
                args.batch_size,
            )

        seq_va = np.array([all_ids[i] for i in va_idx_abs], dtype=object)
        val_path = args.out_dir / "val_probs_for_ensemble.npz"
        _val_kw: dict = {
            "seq_ids": seq_va,
            "p_esm": p_esm_va.astype(np.float32),
            "p_gearnet": p_ge_va.astype(np.float32),
            "y": y_va.astype(np.int64),
        }
        if not two_npz:
            _val_kw["p_saprot"] = p_sap_va.astype(np.float32)
        np.savez(val_path, **_val_kw)
        meta["val_npz"] = str(val_path.resolve())
        meta["val_ratio"] = float(args.val_ratio)
        print("Wrote", val_path)

    if args.phase in ("test", "both"):
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        sum_esm = np.zeros((len(test_ids),), dtype=np.float64)
        sum_sap = np.zeros_like(sum_esm)
        sum_ge = np.zeros_like(sum_esm)
        fill_f = float(args.omit_train_fill)

        for fold, (tr_rel, _) in enumerate(skf.split(np.zeros(len(train_y)), train_y)):
            ge.set_seed(args.seed + fold)
            inner_tr, inner_va = split_fold_train_indices(
                tr_rel, train_y, seed=args.seed + fold * 97, inner_val_ratio=args.inner_val_ratio
            )
            tr_abs = train_idx[inner_tr]
            y_tr = train_y[inner_tr]
            if inner_va.size > 0:
                va_abs = train_idx[inner_va]
                y_va = train_y[inner_va]
            else:
                va_abs = None
                y_va = None

            if "esm" in omit_heads:
                sum_esm += fill_f
            else:
                m_esm = ge.MLPClassifier(esm_dim, hidden_dims, args.dropout).to(device)
                ds_tr = ArrayXYDataset(esm_feat[tr_abs], y_tr)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    va_ld = DataLoader(ArrayXYDataset(esm_feat[va_abs], y_va), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_esm, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=False, crit=crit)
                sum_esm += predict_mlp_probs_tta(
                    m_esm,
                    esm_feat[test_idx],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)

            if "saprot" in omit_heads:
                sum_sap += fill_f
            else:
                m_sap = ge.MLPClassifier(concat_dim, hidden_dims, args.dropout).to(device)
                ds_tr = ArrayXYDataset(concat_esm_sap[tr_abs], y_tr)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    va_ld = DataLoader(ArrayXYDataset(concat_esm_sap[va_abs], y_va), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_sap, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=False, crit=crit)
                sum_sap += predict_mlp_probs_tta(
                    m_sap,
                    concat_esm_sap[test_idx],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)

            if "gearnet" in omit_heads:
                sum_ge += fill_f
            elif args.gearnet_branch == "residual":
                m_r = ge.ResidualGateDualHeadClassifier(
                    esm_dim, gear_dim, hidden_dims, args.dropout, gate_hidden=args.gate_hidden
                ).to(device)
                ds_tr = ge.EsmGearMaskDataset(tr_abs, y_tr, esm_feat, gear_feat, gear_mask)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    ds_va = ge.EsmGearMaskDataset(va_abs, y_va, esm_feat, gear_feat, gear_mask)
                    va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_r, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=True, crit=crit)
                sum_ge += predict_residual_probs_tta(
                    m_r,
                    esm_feat[test_idx],
                    gear_feat[test_idx],
                    gear_mask[test_idx],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)
            elif args.gearnet_branch == "weighted_heads":
                m_w = ge.WeightedDualHeadClassifier(esm_dim, gear_dim, hidden_dims, args.dropout).to(device)
                ds_tr = ge.EsmGearMaskDataset(tr_abs, y_tr, esm_feat, gear_feat, gear_mask)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    ds_va = ge.EsmGearMaskDataset(va_abs, y_va, esm_feat, gear_feat, gear_mask)
                    va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_w, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=True, crit=crit)
                sum_ge += predict_residual_probs_tta(
                    m_w,
                    esm_feat[test_idx],
                    gear_feat[test_idx],
                    gear_mask[test_idx],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)
            else:
                assert sasa_raw is not None and contact_all is not None and lap_all is not None
                sm = float(np.mean(sasa_raw[tr_abs]))
                ss = float(np.std(sasa_raw[tr_abs]) + 1e-6)
                sasa_z = (sasa_raw - sm) / ss
                m_r = mvm.ResidualGateMultiViewMTL(
                    esm_dim,
                    gear_dim,
                    args.lap_dim,
                    hidden_dims,
                    args.dropout,
                    gate_hidden=args.gate_hidden,
                    entropy_center=args.entropy_center,
                    entropy_beta=args.entropy_beta,
                ).to(device)
                ds_tr = mvm.MultiViewDataset(tr_abs, y_tr, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    ds_va = mvm.MultiViewDataset(va_abs, y_va, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
                    va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val_multiview(
                    m_r,
                    tr_ld,
                    va_ld,
                    device,
                    args.epochs,
                    args.lr,
                    args.weight_decay,
                    args.lambda_sasa,
                    args.lambda_ctr,
                    crit,
                )
                sum_ge += predict_multiview_probs_tta(
                    m_r,
                    esm_feat[test_idx],
                    gear_feat[test_idx],
                    gear_mask[test_idx],
                    contact_all[test_idx],
                    lap_all[test_idx],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)

            print(f"[ensemble prep] fold {fold + 1}/{args.n_splits} done")

        inv = 1.0 / float(args.n_splits)
        p_esm_te = (sum_esm * inv).astype(np.float32)
        p_sap_te = (sum_sap * inv).astype(np.float32)
        p_ge_te = (sum_ge * inv).astype(np.float32)
        seq_te = np.array(test_ids, dtype=object)

        test_path = args.out_dir / "test_probs_for_ensemble.npz"
        _te_kw: dict = {
            "seq_ids": seq_te,
            "p_esm": p_esm_te,
            "p_gearnet": p_ge_te,
            "y": test_y.astype(np.int64),
        }
        if not two_npz:
            _te_kw["p_saprot"] = p_sap_te
        np.savez(test_path, **_te_kw)
        meta["test_npz"] = str(test_path.resolve())
        meta["test_cv_splits"] = int(args.n_splits)
        print("Wrote", test_path)

    if args.phase == "train_oof":
        n_oof = max(2, int(args.train_oof_splits))
        skf_oof = StratifiedKFold(n_splits=n_oof, shuffle=True, random_state=int(args.seed) + 777)
        n_tr = int(len(train_y))
        p_esm_oof = np.zeros((n_tr,), dtype=np.float64)
        p_sap_oof = np.zeros_like(p_esm_oof)
        p_ge_oof = np.zeros_like(p_esm_oof)

        for fold, (tr_rel, ho_rel) in enumerate(skf_oof.split(np.zeros(n_tr), train_y)):
            ge.set_seed(int(args.seed) + fold + 333)
            inner_tr, inner_va = split_fold_train_indices(
                tr_rel, train_y, seed=int(args.seed) + fold * 97, inner_val_ratio=args.inner_val_ratio
            )
            tr_abs = train_idx[inner_tr]
            y_tr = train_y[inner_tr]
            if inner_va.size > 0:
                va_abs = train_idx[inner_va]
                y_va = train_y[inner_va]
            else:
                va_abs = None
                y_va = None

            ho_abs = train_idx[ho_rel]
            fill_f = float(args.omit_train_fill)

            if "esm" in omit_heads:
                p_esm_oof[ho_rel] = fill_f
            else:
                m_esm = ge.MLPClassifier(esm_dim, hidden_dims, args.dropout).to(device)
                ds_tr = ArrayXYDataset(esm_feat[tr_abs], y_tr)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    va_ld = DataLoader(ArrayXYDataset(esm_feat[va_abs], y_va), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_esm, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=False, crit=crit)
                p_esm_oof[ho_rel] = predict_mlp_probs_tta(
                    m_esm,
                    esm_feat[ho_abs],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)

            if "saprot" in omit_heads:
                p_sap_oof[ho_rel] = fill_f
            else:
                m_sap = ge.MLPClassifier(concat_dim, hidden_dims, args.dropout).to(device)
                ds_tr = ArrayXYDataset(concat_esm_sap[tr_abs], y_tr)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    va_ld = DataLoader(ArrayXYDataset(concat_esm_sap[va_abs], y_va), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_sap, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=False, crit=crit)
                p_sap_oof[ho_rel] = predict_mlp_probs_tta(
                    m_sap,
                    concat_esm_sap[ho_abs],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)

            if "gearnet" in omit_heads:
                p_ge_oof[ho_rel] = fill_f
            elif args.gearnet_branch == "residual":
                m_r = ge.ResidualGateDualHeadClassifier(
                    esm_dim, gear_dim, hidden_dims, args.dropout, gate_hidden=args.gate_hidden
                ).to(device)
                ds_tr = ge.EsmGearMaskDataset(tr_abs, y_tr, esm_feat, gear_feat, gear_mask)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    ds_va = ge.EsmGearMaskDataset(va_abs, y_va, esm_feat, gear_feat, gear_mask)
                    va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_r, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=True, crit=crit)
                p_ge_oof[ho_rel] = predict_residual_probs_tta(
                    m_r,
                    esm_feat[ho_abs],
                    gear_feat[ho_abs],
                    gear_mask[ho_abs],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)
            elif args.gearnet_branch == "weighted_heads":
                m_w = ge.WeightedDualHeadClassifier(esm_dim, gear_dim, hidden_dims, args.dropout).to(device)
                ds_tr = ge.EsmGearMaskDataset(tr_abs, y_tr, esm_feat, gear_feat, gear_mask)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    ds_va = ge.EsmGearMaskDataset(va_abs, y_va, esm_feat, gear_feat, gear_mask)
                    va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val(m_w, tr_ld, va_ld, device, args.epochs, args.lr, args.weight_decay, dual_head=True, crit=crit)
                p_ge_oof[ho_rel] = predict_residual_probs_tta(
                    m_w,
                    esm_feat[ho_abs],
                    gear_feat[ho_abs],
                    gear_mask[ho_abs],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)
            else:
                assert sasa_raw is not None and contact_all is not None and lap_all is not None
                sm = float(np.mean(sasa_raw[tr_abs]))
                ss = float(np.std(sasa_raw[tr_abs]) + 1e-6)
                sasa_z = (sasa_raw - sm) / ss
                m_r = mvm.ResidualGateMultiViewMTL(
                    esm_dim,
                    gear_dim,
                    args.lap_dim,
                    hidden_dims,
                    args.dropout,
                    gate_hidden=args.gate_hidden,
                    entropy_center=args.entropy_center,
                    entropy_beta=args.entropy_beta,
                ).to(device)
                ds_tr = mvm.MultiViewDataset(tr_abs, y_tr, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
                tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
                if va_abs is not None:
                    ds_va = mvm.MultiViewDataset(va_abs, y_va, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
                    va_ld = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
                else:
                    va_ld = None
                train_until_best_val_multiview(
                    m_r,
                    tr_ld,
                    va_ld,
                    device,
                    args.epochs,
                    args.lr,
                    args.weight_decay,
                    args.lambda_sasa,
                    args.lambda_ctr,
                    crit,
                )
                p_ge_oof[ho_rel] = predict_multiview_probs_tta(
                    m_r,
                    esm_feat[ho_abs],
                    gear_feat[ho_abs],
                    gear_mask[ho_abs],
                    contact_all[ho_abs],
                    lap_all[ho_abs],
                    device,
                    args.batch_size,
                    tta_runs=args.tta_runs,
                    noise_std=args.tta_noise_std,
                ).astype(np.float64)

            print(f"[ensemble prep train_oof] fold {fold + 1}/{n_oof} done")

        seq_tr = np.array(train_ids, dtype=object)
        oof_path = args.out_dir / "train_oof_probs_for_ensemble.npz"
        _oof_kw: dict = {
            "seq_ids": seq_tr,
            "p_esm": p_esm_oof.astype(np.float32),
            "p_gearnet": p_ge_oof.astype(np.float32),
            "y": train_y.astype(np.int64),
        }
        if not two_npz:
            _oof_kw["p_saprot"] = p_sap_oof.astype(np.float32)
        np.savez(oof_path, **_oof_kw)
        meta["train_oof_npz"] = str(oof_path.resolve())
        meta["train_oof_splits"] = int(n_oof)
        print("Wrote", oof_path)

    meta["tta_runs"] = int(args.tta_runs)
    meta["tta_noise_std"] = float(args.tta_noise_std)
    meta_path = args.out_dir / "prepare_ensemble_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", meta_path)
    print("\n下一步示例（NPZ 键名不变；four_panel 及 feat CSV 等请与 B 线一致）：")
    if meta.get("val_npz") and meta.get("test_npz"):
        print(
            "python3 classifier_scripts/ensemble_stacking_plus_abc.py --four_panel "
            f"--val_npz {args.out_dir / 'val_probs_for_ensemble.npz'} "
            f"--test_npz {args.out_dir / 'test_probs_for_ensemble.npz'} "
            "--out_json trained/stacking_result.json"
        )
    if meta.get("train_oof_npz"):
        print("train OOF 三路概率（可与 val 拼接后训二级 Meta，防 val 过拟合）：", meta["train_oof_npz"])


if __name__ == "__main__":
    main()
