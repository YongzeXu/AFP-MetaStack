#!/usr/bin/env python3
"""
预提取 AFP-MetaStack 全量样本的 ESM2 / SaProt token-bin 特征（与 train_esm2_plus_saprot_token_crossattn_cv10 池化逻辑一致），
保存到磁盘供全离线训练只训 Cross-Attn+MLP。

输出目录结构（默认 features/afp_e_esm2_saprot_token_bins；改 token_bins 时请用 --out_dir 指到新目录，例如 afp_e_esm2_saprot_token_bins_16）：
  meta.json           — 维度、模型路径、参数等
  seq_ids.json        — 与 *.npy 行顺序一致的 seq_id 列表
  labels.json         — seq_id -> 训练/测试标签（若可在 splits 中解析）
  esm2_token_bins.npy — float16, [N, token_bins, D_esm2]
  saprot_token_bins.npy — float16, [N, token_bins, D_saprot]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _models_root() -> Path:
    import os
    return Path(os.environ.get("AFP_MODELS_ROOT", Path.home() / "models"))
DEFAULT_MISSING_TOKEN = "#"


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


def load_3di_map(three_di_fasta: Path) -> Dict[str, str]:
    raw: Dict[str, str] = {}
    sid = ""
    seq_parts: List[str] = []
    with three_di_fasta.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid:
                    s = "".join(seq_parts)
                    if s:
                        raw[sid] = s
                sid = line[1:].strip().split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if sid:
        s = "".join(seq_parts)
        if s:
            raw[sid] = s

    best: Dict[str, str] = {}
    for k, v in raw.items():
        parts = k.split("_")
        sid2 = "_".join(parts[:3]) if len(parts) >= 3 and parts[0] in {"P", "N"} and parts[1] == "seq" else k
        if len(v) > len(best.get(sid2, "")):
            best[sid2] = v
    return best


def make_saprot_input(seq: str, three_di: str, missing_token: str) -> str:
    out: List[str] = []
    for i, aa in enumerate(seq):
        sa = three_di[i] if i < len(three_di) else missing_token
        out.append(aa)
        out.append(sa)
    return "".join(out)


@torch.no_grad()
def build_token_embeddings(
    seq_ids: List[str],
    text_map: Dict[str, str],
    model_path: Path,
    batch_size: int,
    max_length: int,
    token_bins: int,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    tokenizer = EsmTokenizer.from_pretrained(str(model_path))
    model = EsmModel.from_pretrained(str(model_path)).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dim: Optional[int] = None
    out_buf: Optional[np.ndarray] = None

    for i in range(0, len(seq_ids), batch_size):
        chunk = seq_ids[i : i + batch_size]
        texts = [text_map[sid] for sid in chunk]
        toks = tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attn = toks["attention_mask"]
        toks = {k: v.to(device) for k, v in toks.items()}
        hs = model(**toks).last_hidden_state
        hs_cpu = hs.detach().cpu()
        attn_cpu = attn.detach().cpu()

        if dim is None:
            dim = int(hs_cpu.shape[-1])
            out_buf = np.zeros((len(seq_ids), token_bins, dim), dtype=np.float16)

        assert out_buf is not None and dim is not None
        for j in range(hs_cpu.shape[0]):
            valid = int(attn_cpu[j].sum().item())
            l0 = 1
            r0 = max(l0 + 1, valid - 1)
            token_h = hs_cpu[j, l0:r0, :]
            if token_h.shape[0] == 0:
                token_h = hs_cpu[j, :1, :]
            pooled = F.adaptive_avg_pool1d(token_h.T.unsqueeze(0), token_bins).squeeze(0).T
            out_buf[i + j] = pooled.numpy().astype(np.float16)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert out_buf is not None and dim is not None
    return out_buf, dim


@torch.no_grad()
def build_token_embeddings_multilayer(
    seq_ids: List[str],
    text_map: Dict[str, str],
    model_path: Path,
    batch_size: int,
    max_length: int,
    token_bins: int,
    device: torch.device,
    repr_layers: List[int],
    layer_weights: List[float],
) -> Tuple[np.ndarray, int]:
    """
    从同一模型多个层取 hidden states，并按给定权重做加权聚合，再走与单层一致的 token-bin pooling。
    """
    if len(repr_layers) != len(layer_weights):
        raise ValueError("repr_layers 与 layer_weights 长度必须一致")
    if len(repr_layers) == 0:
        raise ValueError("repr_layers 不能为空")
    w = np.array(layer_weights, dtype=np.float64)
    if float(np.abs(w).sum()) <= 0:
        raise ValueError("layer_weights 不能全为 0")
    w = (w / np.abs(w).sum()).astype(np.float32)

    tokenizer = EsmTokenizer.from_pretrained(str(model_path))
    model = EsmModel.from_pretrained(str(model_path)).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dim: Optional[int] = None
    out_buf: Optional[np.ndarray] = None

    for i in range(0, len(seq_ids), batch_size):
        chunk = seq_ids[i : i + batch_size]
        texts = [text_map[sid] for sid in chunk]
        toks = tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attn = toks["attention_mask"]
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks, output_hidden_states=True, return_dict=True)
        hs_all = out.hidden_states
        if hs_all is None:
            raise RuntimeError("模型未返回 hidden_states")
        # hidden_states: tuple length = num_layers + 1（含 embedding 输出）
        max_idx = len(hs_all) - 1
        idxs: List[int] = []
        for ly in repr_layers:
            if ly < 0 or ly > max_idx:
                raise ValueError(f"层索引 {ly} 超出范围 [0, {max_idx}]")
            idxs.append(int(ly))
        hs_mix = None
        for k, idx in enumerate(idxs):
            h = hs_all[idx]
            hs_mix = h * w[k] if hs_mix is None else (hs_mix + h * w[k])
        assert hs_mix is not None
        hs_cpu = hs_mix.detach().cpu()
        attn_cpu = attn.detach().cpu()

        if dim is None:
            dim = int(hs_cpu.shape[-1])
            out_buf = np.zeros((len(seq_ids), token_bins, dim), dtype=np.float16)

        assert out_buf is not None and dim is not None
        for j in range(hs_cpu.shape[0]):
            valid = int(attn_cpu[j].sum().item())
            l0 = 1
            r0 = max(l0 + 1, valid - 1)
            token_h = hs_cpu[j, l0:r0, :]
            if token_h.shape[0] == 0:
                token_h = hs_cpu[j, :1, :]
            pooled = F.adaptive_avg_pool1d(token_h.T.unsqueeze(0), token_bins).squeeze(0).T
            out_buf[i + j] = pooled.numpy().astype(np.float16)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert out_buf is not None and dim is not None
    return out_buf, dim


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="提取 AFP-MetaStack ESM2 + SaProt token-bin 特征并落盘")
    p.add_argument("--train_pos_fasta", type=Path, default=ROOT / "data" / "train_set" / "pos_train.fasta")
    p.add_argument("--train_neg_fasta", type=Path, default=ROOT / "data" / "train_set" / "neg_train.fasta")
    p.add_argument("--test_pos_fasta", type=Path, default=ROOT / "data" / "test_set" / "pos_test.fasta")
    p.add_argument("--test_neg_fasta", type=Path, default=ROOT / "data" / "test_set" / "neg_test.fasta")
    p.add_argument("--train_3di_fasta", type=Path, default=ROOT / "data" / "train_set" / "3di" / "train_3di.fasta")
    p.add_argument("--test_3di_fasta", type=Path, default=ROOT / "data" / "test_set" / "3di" / "test_3di.fasta")
    p.add_argument("--esm2_model_dir", type=Path, default=_models_root() / "facebook/esm2_t33_650M_UR50D")
    p.add_argument("--saprot_model_dir", type=Path, default=_models_root() / "westlake-repl/SaProt_650M_AF2")
    p.add_argument("--max_length_esm2", type=int, default=1022)
    p.add_argument("--max_length_saprot", type=int, default=2046)
    p.add_argument("--token_bins", type=int, default=8)
    p.add_argument(
        "--missing_token",
        type=str,
        default=DEFAULT_MISSING_TOKEN,
        help="SaProt 结构缺失位的占位 token。建议用更中性的字符（例如 X），以避免 # 噪声占比过高。",
    )
    p.add_argument("--embed_batch_size", type=int, default=8)
    p.add_argument(
        "--esm2_repr_layers",
        type=str,
        default="",
        help="可选：多层聚合层索引，逗号分隔，如 '12,24,33'；留空表示仅最后层（原行为）",
    )
    p.add_argument(
        "--esm2_layer_weights",
        type=str,
        default="",
        help="可选：与 esm2_repr_layers 对应的权重，如 '0.2,0.3,0.5'；留空则均匀权重",
    )
    p.add_argument("--out_dir", type=Path, default=ROOT / "features" / "afp_e_esm2_saprot_token_bins")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = load_split(args.train_pos_fasta.resolve(), args.train_neg_fasta.resolve())
    test_rows = load_split(args.test_pos_fasta.resolve(), args.test_neg_fasta.resolve())
    train_3di = load_3di_map(args.train_3di_fasta.resolve())
    test_3di = load_3di_map(args.test_3di_fasta.resolve())

    train_ids = [x[0] for x in train_rows]
    test_ids = [x[0] for x in test_rows]
    seq_map = {sid: seq for sid, _, seq in (train_rows + test_rows)}
    label_map: Dict[str, int] = {}
    for sid, lab, _ in train_rows:
        label_map[sid] = lab
    for sid, lab, _ in test_rows:
        label_map[sid] = lab

    all_ids = list(dict.fromkeys(train_ids + test_ids))
    all_3di: Dict[str, str] = {}
    all_3di.update(train_3di)
    all_3di.update(test_3di)
    saprot_input_map = {
        sid: make_saprot_input(seq_map[sid], all_3di.get(sid, ""), args.missing_token) for sid in all_ids
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("N samples:", len(all_ids))
    print("token_bins:", args.token_bins)
    print("Writing to:", args.out_dir.resolve())

    esm2_layers = _parse_int_list(args.esm2_repr_layers) if args.esm2_repr_layers.strip() else []
    if len(esm2_layers) > 0:
        if args.esm2_layer_weights.strip():
            esm2_w = _parse_float_list(args.esm2_layer_weights)
        else:
            esm2_w = [1.0] * len(esm2_layers)
        print("ESM2 multi-layer aggregation enabled:", esm2_layers, "weights=", esm2_w)
        emb_esm2, dim_esm2 = build_token_embeddings_multilayer(
            seq_ids=all_ids,
            text_map=seq_map,
            model_path=args.esm2_model_dir.resolve(),
            batch_size=args.embed_batch_size,
            max_length=args.max_length_esm2,
            token_bins=args.token_bins,
            device=device,
            repr_layers=esm2_layers,
            layer_weights=esm2_w,
        )
    else:
        emb_esm2, dim_esm2 = build_token_embeddings(
            seq_ids=all_ids,
            text_map=seq_map,
            model_path=args.esm2_model_dir.resolve(),
            batch_size=args.embed_batch_size,
            max_length=args.max_length_esm2,
            token_bins=args.token_bins,
            device=device,
        )
    emb_sap, dim_sap = build_token_embeddings(
        seq_ids=all_ids,
        text_map=saprot_input_map,
        model_path=args.saprot_model_dir.resolve(),
        batch_size=args.embed_batch_size,
        max_length=args.max_length_saprot,
        token_bins=args.token_bins,
        device=device,
    )

    np.save(args.out_dir / "esm2_token_bins.npy", emb_esm2)
    np.save(args.out_dir / "saprot_token_bins.npy", emb_sap)

    labels_list = [int(label_map[sid]) for sid in all_ids]
    with (args.out_dir / "seq_ids.json").open("w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)
    with (args.out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({sid: label_map[sid] for sid in all_ids}, f, ensure_ascii=False, indent=2)

    meta = {
        "n_samples": len(all_ids),
        "token_bins": args.token_bins,
        "missing_token": args.missing_token,
        "dim_esm2": dim_esm2,
        "dim_saprot": dim_sap,
        "dtype_stored": "float16",
        "max_length_esm2": args.max_length_esm2,
        "max_length_saprot": args.max_length_saprot,
        "esm2_model_dir": str(args.esm2_model_dir.resolve()),
        "saprot_model_dir": str(args.saprot_model_dir.resolve()),
        "train_pos_fasta": str(args.train_pos_fasta.resolve()),
        "train_neg_fasta": str(args.train_neg_fasta.resolve()),
        "test_pos_fasta": str(args.test_pos_fasta.resolve()),
        "test_neg_fasta": str(args.test_neg_fasta.resolve()),
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "note": "行顺序与 seq_ids.json 一致；与 train_esm2_plus_saprot_token_crossattn_cv10 池化一致",
        "esm2_repr_layers": esm2_layers if len(esm2_layers) > 0 else "last_hidden_state_only",
        "esm2_layer_weights": args.esm2_layer_weights if args.esm2_layer_weights.strip() else "uniform_or_default",
    }
    with (args.out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    np.save(args.out_dir / "labels_per_row.npy", np.array(labels_list, dtype=np.int64))

    print("saved:", args.out_dir / "esm2_token_bins.npy", emb_esm2.shape, emb_esm2.dtype)
    print("saved:", args.out_dir / "saprot_token_bins.npy", emb_sap.shape, emb_sap.dtype)
    print("done.")


if __name__ == "__main__":
    main()
