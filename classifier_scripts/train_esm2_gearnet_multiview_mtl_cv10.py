#!/usr/bin/env python3
"""
ESM2(mean-pool) + GearNet-MC + 残差门控双头 的扩展版（一次集成多张「图」里的思路）：

1) 结构多视角：GearNet 向量 + Contact/Distance map（轻量 CNN）+ Cα KNN 拉普拉斯谱（线性投影）
2) 多任务：回归 PDB 派生的 sasa_proxy（质心距离均值，粗暴露度代理）
3) 视角对齐：GearNet 与 contact 嵌入的余弦一致性损失（轻量「对比」）
4) 推理可选：按 ESM 头一元熵抬高结构支路概率权重（熵动态仲裁近似）

用法示例：
  python features_scripts/experimental/build_struct_multiview_cache.py
  python classifier_scripts/train_esm2_gearnet_multiview_mtl_cv10.py \\
      --struct_aux_cache_dir /data/yongze/AFP-MetaStack/features/afp_e_struct_multiview

说明：IBS 精细标注、真实 SASA（freesasa）、MSA 保守性未包含——需额外标注/流水线。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


def _models_root() -> Path:
    import os
    return Path(os.environ.get("AFP_MODELS_ROOT", Path.home() / "models"))


def line_buffer_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass


def _load_base_cv_module():
    p = Path(__file__).resolve().parent / "train_esm2_plus_gearnet_concat_cv10.py"
    spec = importlib.util.spec_from_file_location("esm_gearnet_cv10", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}\n>>> 本脚本依赖与 train_esm2_plus_gearnet_concat_cv10 相同的环境（含 torchdrug）。"
            "请勿用系统裸 python 跑 nohup：先 `conda activate` 装好了 torchdrug 的环境，再执行 python。"
        ) from e
    return mod


class MultiViewDataset(Dataset):
    def __init__(
        self,
        row_idx: np.ndarray,
        labels: np.ndarray,
        esm_feat: np.ndarray,
        gear_feat: np.ndarray,
        gear_mask: np.ndarray,
        contact: np.ndarray,
        lap: np.ndarray,
        sasa_tgt: np.ndarray,
    ):
        self.row_idx = row_idx.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.esm_feat = esm_feat
        self.gear_feat = gear_feat
        self.gear_mask = gear_mask.astype(np.float32)
        self.contact = contact.astype(np.float32)
        self.lap = lap.astype(np.float32)
        self.sasa_tgt = sasa_tgt.astype(np.float32)

    def __len__(self) -> int:
        return int(self.row_idx.shape[0])

    def __getitem__(self, idx: int):
        i = int(self.row_idx[idx])
        e = torch.from_numpy(self.esm_feat[i])
        g = torch.from_numpy(self.gear_feat[i])
        m = torch.tensor(float(self.gear_mask[i]), dtype=torch.float32)
        c = torch.from_numpy(self.contact[i]).unsqueeze(0)
        lap = torch.from_numpy(self.lap[i])
        sasa = torch.tensor(float(self.sasa_tgt[i]), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return e, g, m, c, lap, sasa, y


class ResidualGateMultiViewMTL(nn.Module):
    """
    logits = logits_esm + λ·logits_gear，λ=mask·σ(Gate)；
    gear 分支输入 = GearNet ⊕ CNN(contact) ⊕ MLP(Lap)。
    """

    def __init__(
        self,
        esm_dim: int,
        gear_dim: int,
        lap_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
        gate_hidden: int,
        ctr_dim: int = 64,
        lap_proj_dim: int = 32,
        entropy_center: float = 0.35,
        entropy_beta: float = 8.0,
    ):
        super().__init__()
        self.gear_dim = gear_dim
        self.entropy_center = entropy_center
        self.entropy_beta = entropy_beta

        self.contact_enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, ctr_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.lap_mlp = nn.Sequential(nn.Linear(lap_dim, lap_proj_dim), nn.ReLU(), nn.Dropout(dropout))

        aug_dim = gear_dim + ctr_dim + lap_proj_dim

        def _mlp_head(in_dim: int) -> nn.Sequential:
            layers_: List[nn.Module] = []
            d = in_dim
            for h in hidden_dims:
                layers_.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
                d = h
            layers_.append(nn.Linear(d, 2))
            return nn.Sequential(*layers_)

        self.head_esm = _mlp_head(esm_dim)
        self.head_gear = _mlp_head(aug_dim)
        gh = max(8, int(gate_hidden))
        self.gate = nn.Sequential(
            nn.Linear(esm_dim + aug_dim + 1, gh),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gh, 1),
        )
        self.sasa_head = nn.Sequential(
            nn.Linear(esm_dim + aug_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.proj_g = nn.Linear(gear_dim, ctr_dim)
        self.proj_c = nn.Linear(ctr_dim, ctr_dim)

    def encode_aug(self, gear: torch.Tensor, mask: torch.Tensor, contact: torch.Tensor, lap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_c = self.contact_enc(contact)
        z_l = self.lap_mlp(lap)
        if mask.dim() == 1:
            m = mask.unsqueeze(-1)
        else:
            m = mask
        m = m.clamp(0.0, 1.0)
        z_c = z_c * m
        z_l = z_l * m
        aug = torch.cat([gear, z_c, z_l], dim=-1)
        return aug, z_c

    def forward(
        self,
        esm: torch.Tensor,
        gear: torch.Tensor,
        mask: torch.Tensor,
        contact: torch.Tensor,
        lap: torch.Tensor,
        entropy_blend_infer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        aug, z_c = self.encode_aug(gear, mask, contact, lap)
        le = self.head_esm(esm)
        lg = self.head_gear(aug)
        if mask.dim() == 1:
            m = mask.unsqueeze(-1)
        else:
            m = mask
        m = m.clamp(0.0, 1.0).to(dtype=esm.dtype)
        gate_in = torch.cat([esm, aug, m], dim=-1)
        lam = torch.sigmoid(self.gate(gate_in)) * m
        logits = le + lam * lg

        aux: Dict[str, torch.Tensor] = {
            "le": le,
            "lg": lg,
            "lam": lam,
            "z_c": z_c,
            "gear": gear,
            "aug": aug,
        }

        if entropy_blend_infer:
            pe = F.softmax(le, dim=1)[:, 1]
            pg = F.softmax(lg, dim=1)[:, 1]
            ent = -(pe * torch.log(pe + 1e-8) + (1.0 - pe) * torch.log(1.0 - pe + 1e-8))
            w = m.squeeze(-1) * torch.sigmoid((ent - self.entropy_center) * self.entropy_beta)
            p_mix = (1.0 - w) * pe + w * pg
            logits = torch.stack([torch.log(1.0 - p_mix + 1e-8), torch.log(p_mix + 1e-8)], dim=1)

        return logits, self.sasa_head(torch.cat([esm, aug], dim=-1)).squeeze(-1), aux


def run_epoch_multiview(
    model: ResidualGateMultiViewMTL,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    lambda_sasa: float,
    lambda_ctr: float,
    train: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    entropy_infer: bool = False,
    metrics_fn=None,
) -> Dict[str, float]:
    model.train(train)
    total = 0.0
    n_batches = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []

    with torch.set_grad_enabled(train):
        for batch in loader:
            e, g, mask, contact, lap, sasa_y, y = batch
            e = e.to(device)
            g = g.to(device)
            mask = mask.to(device)
            contact = contact.to(device)
            lap = lap.to(device)
            sasa_y = sasa_y.to(device)
            y = y.to(device)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            logits, sasa_pred, aux = model(e, g, mask, contact, lap, entropy_blend_infer=entropy_infer and not train)
            loss = criterion(logits, y)
            if lambda_sasa > 0.0:
                msk = mask > 0.5
                if msk.any():
                    loss = loss + lambda_sasa * F.mse_loss(sasa_pred[msk], sasa_y[msk])
            if lambda_ctr > 0.0 and train:
                sel = mask.squeeze(-1) > 0.5
                if sel.any():
                    zg = F.normalize(model.proj_g(aux["gear"][sel]), dim=1)
                    zc = F.normalize(model.proj_c(aux["z_c"][sel]), dim=1)
                    loss_ctr = (1.0 - (zg * zc).sum(dim=1)).mean()
                    loss = loss + lambda_ctr * loss_ctr
            if train:
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            n_batches += 1
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())

    assert metrics_fn is not None
    m = metrics_fn(np.array(all_true), np.array(all_pred), np.array(all_prob))
    m["loss"] = total / max(n_batches, 1)
    return m


def align_multiview_arrays(all_ids: List[str], cache_dir: Path, lap_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid_path = cache_dir / "seq_ids.json"
    if not sid_path.is_file():
        raise FileNotFoundError(
            f"缺少 {sid_path}，请先运行 features_scripts/experimental/build_struct_multiview_cache.py"
        )
    with sid_path.open("r", encoding="utf-8") as f:
        cached_ids: List[str] = json.load(f)
    row_of = {s: i for i, s in enumerate(cached_ids)}
    contact_path = cache_dir / "contact_L256.npy"
    lap_path = cache_dir / f"laplacian_k{lap_dim}.npy"
    sasa_path = cache_dir / "sasa_proxy.npy"
    if not contact_path.is_file() or not lap_path.is_file() or not sasa_path.is_file():
        raise FileNotFoundError(f"缓存不完整：{contact_path}, {lap_path}, {sasa_path}")
    contact_all = np.load(contact_path)
    lap_all = np.load(lap_path)
    sasa_all = np.load(sasa_path)

    n = len(all_ids)
    contact = np.zeros((n,) + tuple(contact_all.shape[1:]), dtype=np.float32)
    lap = np.zeros((n, lap_dim), dtype=np.float32)
    sasa = np.zeros((n,), dtype=np.float32)
    for i, sid in enumerate(all_ids):
        j = row_of.get(sid)
        if j is None:
            continue
        contact[i] = contact_all[j]
        lap[i] = lap_all[j]
        sasa[i] = sasa_all[j]
    return contact, lap, sasa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ESM2+GearNet 残差门控 + 多视角结构 + MTL + 熵推理")
    p.add_argument("--train_pos_fasta", type=Path, default=ROOT / "data" / "train_set" / "pos_train.fasta")
    p.add_argument("--train_neg_fasta", type=Path, default=ROOT / "data" / "train_set" / "neg_train.fasta")
    p.add_argument("--test_pos_fasta", type=Path, default=ROOT / "data" / "test_set" / "pos_test.fasta")
    p.add_argument("--test_neg_fasta", type=Path, default=ROOT / "data" / "test_set" / "neg_test.fasta")
    p.add_argument("--esm2_cache_dir", type=Path, default=ROOT / "features" / "afp_e_esm2_saprot_token_bins")
    p.add_argument("--gearnet_checkpoint", type=Path, default=_models_root() / "gearnet/checkpoints/mc_gearnet_edge.pth")
    p.add_argument("--pdb_dir", type=Path, default=ROOT / "data" / "raw" / "pdb")
    p.add_argument("--gearnet_emb_cache_dir", type=Path, default=ROOT / "features" / "afp_e_gearnet_mc_emb_cache")
    p.add_argument("--struct_aux_cache_dir", type=Path, default=ROOT / "features" / "afp_e_struct_multiview")
    p.add_argument("--hidden_dims", type=str, default="512,256")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--lap_dim", type=int, default=16)
    p.add_argument("--lambda_sasa", type=float, default=0.05, help="SASA 代理回归权重")
    p.add_argument("--lambda_ctr", type=float, default=0.02, help="GearNet–contact 对齐损失权重")
    p.add_argument("--entropy_center", type=float, default=0.35)
    p.add_argument("--entropy_beta", type=float, default=8.0)
    p.add_argument("--test_entropy_blend", action="store_true", help="测试集使用熵动态混合概率（验证仍关闭以便选 checkpoint）")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=Path, default=ROOT / "trained" / "esm2_gearnet_multiview_mtl_cv10")
    p.add_argument("--experiments_csv", type=Path, default=ROOT / "results" / "experiments_table_zh.csv")
    return p.parse_args()


def main() -> None:
    line_buffer_stdout()
    args = parse_args()
    tg = _load_base_cv_module()
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())
    if not hidden_dims:
        raise ValueError("hidden_dims 至少包含一层")

    train_rows = tg.load_split(args.train_pos_fasta.resolve(), args.train_neg_fasta.resolve())
    test_rows = tg.load_split(args.test_pos_fasta.resolve(), args.test_neg_fasta.resolve())
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
    tg.set_seed(args.seed)
    print("device:", device)

    esm_feat, esm_dim = tg.load_esm2_from_cache(args.esm2_cache_dir.resolve(), all_ids)
    gear_feat, gear_mask, gear_dim = tg.build_or_load_gearnet_embeddings(
        all_ids=all_ids,
        pdb_dir=args.pdb_dir.resolve(),
        checkpoint=args.gearnet_checkpoint.resolve(),
        cache_dir=args.gearnet_emb_cache_dir.resolve(),
        device=device,
    )
    contact_all, lap_all, sasa_raw = align_multiview_arrays(all_ids, args.struct_aux_cache_dir.resolve(), args.lap_dim)

    # SASA 代理：按训练折标准化（此处先用全 train 统计近似；每折可再细化）
    tr_vals = sasa_raw[train_idx]
    sm = float(tr_vals.mean())
    ss = float(tr_vals.std() + 1e-6)
    sasa_z = (sasa_raw - sm) / ss

    folds: List[Dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (tr_i, va_i) in enumerate(skf.split(np.zeros(len(train_y)), train_y)):
        tg.set_seed(args.seed + fold)
        tr_row = train_idx[tr_i]
        va_row = train_idx[va_i]
        tr_y = train_y[tr_i]
        va_y = train_y[va_i]

        tr_ds = MultiViewDataset(tr_row, tr_y, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
        va_ds = MultiViewDataset(va_row, va_y, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)
        te_ds = MultiViewDataset(test_idx, test_y, esm_feat, gear_feat, gear_mask, contact_all, lap_all, sasa_z)

        pin = device.type == "cuda"
        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
        va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

        model = ResidualGateMultiViewMTL(
            esm_dim=esm_dim,
            gear_dim=gear_dim,
            lap_dim=args.lap_dim,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            gate_hidden=args.gate_hidden,
            entropy_center=args.entropy_center,
            entropy_beta=args.entropy_beta,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None

        print(f"\n=== Fold {fold + 1}/{args.n_splits} multiview_mtl ===", flush=True)
        for ep in range(args.epochs):
            tr_m = run_epoch_multiview(
                model,
                tr_ld,
                device,
                criterion,
                args.lambda_sasa,
                args.lambda_ctr,
                train=True,
                optimizer=optimizer,
                entropy_infer=False,
                metrics_fn=tg._metrics_block,
            )
            va_m = run_epoch_multiview(
                model,
                va_ld,
                device,
                criterion,
                args.lambda_sasa,
                args.lambda_ctr,
                train=False,
                entropy_infer=False,
                metrics_fn=tg._metrics_block,
            )
            print(
                f"  Epoch {ep + 1}/{args.epochs} train_loss={tr_m['loss']:.4f} val_loss={va_m['loss']:.4f} val_acc={va_m['accuracy']:.4f}",
                flush=True,
            )
            if float(va_m["loss"]) < best_val:
                best_val = float(va_m["loss"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("no best state")
        model.load_state_dict(best_state)
        fs: Dict[str, Any] = {"fold": fold}
        fs["val"] = run_epoch_multiview(
            model, va_ld, device, criterion, args.lambda_sasa, args.lambda_ctr, train=False, entropy_infer=False, metrics_fn=tg._metrics_block
        )
        fs["val"].pop("loss", None)
        fs["test_plain"] = run_epoch_multiview(
            model, te_ld, device, criterion, args.lambda_sasa, args.lambda_ctr, train=False, entropy_infer=False, metrics_fn=tg._metrics_block
        )
        fs["test_plain"].pop("loss", None)
        if args.test_entropy_blend:
            fs["test_entropy"] = run_epoch_multiview(
                model,
                te_ld,
                device,
                criterion,
                args.lambda_sasa,
                args.lambda_ctr,
                train=False,
                entropy_infer=True,
                metrics_fn=tg._metrics_block,
            )
            fs["test_entropy"].pop("loss", None)
        folds.append(fs)

    report: Dict[str, Any] = {
        "method": "ESM2_plus_GearNet_multiview_mtl_residual_gate",
        "struct_aux_cache_dir": str(args.struct_aux_cache_dir.resolve()),
        "lambda_sasa": args.lambda_sasa,
        "lambda_ctr": args.lambda_ctr,
        "lap_dim": args.lap_dim,
        "entropy_center": args.entropy_center,
        "entropy_beta": args.entropy_beta,
        "test_entropy_blend_eval": bool(args.test_entropy_blend),
        "folds": folds,
        "aggregate_test": {
            "test_accuracy": tg._agg_numeric(folds, ["test_plain", "accuracy"]),
            "test_mcc": tg._agg_numeric(folds, ["test_plain", "mcc"]),
            "test_auc": tg._agg_numeric(folds, ["test_plain", "auc"]),
            "test_aupr": tg._agg_numeric(folds, ["test_plain", "aupr"]),
            "test_sensitivity": tg._agg_numeric(folds, ["test_plain", "sensitivity"]),
            "test_recall": tg._agg_numeric(folds, ["test_plain", "recall"]),
            "test_specificity": tg._agg_numeric(folds, ["test_plain", "specificity"]),
            "test_f1": tg._agg_numeric(folds, ["test_plain", "f1"]),
        },
    }
    if args.test_entropy_blend:
        report["aggregate_test_entropy"] = {
            "test_accuracy": tg._agg_numeric(folds, ["test_entropy", "accuracy"]),
            "test_mcc": tg._agg_numeric(folds, ["test_entropy", "mcc"]),
        }

    out_json = args.out_dir / "cv_metrics_esm2_gearnet_multiview_mtl_on_afp_e_cv10.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"Wrote {out_json}", flush=True)

    at = report["aggregate_test"]
    zh_row = {
        "特征": "ESM2-t33-650M(mean-pool)+GearNet-MC+Contact/Lap多视角+SASA代理MTL+对比对齐",
        "分类器": f"残差门控双头+MTL（λ_sasa={args.lambda_sasa}, λ_ctr={args.lambda_ctr}）",
        "融合器": f"logits_ESM+λ·logits_Gear(aug)；aug=Gear⊕CNN(contact)⊕Lap谱；测试熵混合={'开' if args.test_entropy_blend else '关'}",
        "结构分支": "PDB：GearNet图+256×距离图+拉普拉斯k维；无结构零回退",
        "test_accuracy": f"{at['test_accuracy']['mean']:.4f}",
        "test_mcc": f"{at['test_mcc']['mean']:.4f}",
        "test_auc": f"{at['test_auc']['mean']:.4f}",
        "test_aupr": f"{at['test_aupr']['mean']:.4f}",
        "test_sensitivity": f"{at['test_sensitivity']['mean']:.4f}",
        "test_recall": f"{at['test_recall']['mean']:.4f}",
        "test_specificity": f"{at['test_specificity']['mean']:.4f}",
        "test_f1": f"{at['test_f1']['mean']:.4f}",
    }
    tg.append_experiments_table_row(args.experiments_csv.resolve(), zh_row)
    print(f"Appended row to {args.experiments_csv.resolve()}", flush=True)


if __name__ == "__main__":
    main()
