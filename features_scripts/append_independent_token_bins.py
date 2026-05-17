#!/usr/bin/env python3
"""
在**不**重算全库的前提下，把独立集序列追加进现有 ESM2/SaProt token-bin 缓存目录。

前提：
  - 已运行 processing/prepare_independent_split.py，得到 pos/neg_independent.fasta（INDP*/INDN* id）；
  - 主缓存 meta.json 为「单层 ESM last_hidden」（与当前 features/afp_e_esm2_saprot_token_bins 一致）；
  - SaProt 输入用「无 3Di」：make_saprot_input(seq, \"\", missing_token)，与无结构占位一致。

用法：
  cd /data/yongze/AFP-MetaStack && python3 features_scripts/append_independent_token_bins.py
  # 先 dry-run 看将追加多少条：
  python3 features_scripts/append_independent_token_bins.py --dry_run

建议：运行前自行备份 out_dir 下 *.npy 与 seq_ids.json。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _models_root() -> Path:
    import os
    return Path(os.environ.get("AFP_MODELS_ROOT", Path.home() / "models"))


def _load_ext():
    path = ROOT / "features_scripts" / "extract_esm2_saprot_token_bins.py"
    spec = importlib.util.spec_from_file_location("extract_esm2_saprot_token_bins", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 extract_esm2_saprot_token_bins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将 independent fasta 追加到现有 token-bin 特征缓存")
    p.add_argument("--out_dir", type=Path, default=ROOT / "features" / "afp_e_esm2_saprot_token_bins")
    p.add_argument(
        "--indep_pos",
        type=Path,
        default=ROOT / "data" / "independent_set" / "pos_independent.fasta",
    )
    p.add_argument(
        "--indep_neg",
        type=Path,
        default=ROOT / "data" / "independent_set" / "neg_independent.fasta",
    )
    p.add_argument("--esm2_model_dir", type=Path, default=_models_root() / "facebook/esm2_t33_650M_UR50D")
    p.add_argument("--saprot_model_dir", type=Path, default=_models_root() / "westlake-repl/SaProt_650M_AF2")
    p.add_argument("--max_length_esm2", type=int, default=1022)
    p.add_argument("--max_length_saprot", type=int, default=2046)
    p.add_argument("--embed_batch_size", type=int, default=8)
    p.add_argument("--missing_token", type=str, default="#")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--no_backup",
        action="store_true",
        help="默认会在 out_dir 下写 *.bak.<时间戳>；加此 flag 则跳过备份（不推荐）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out_dir.resolve()
    ext = _load_ext()
    seq_path = out / "seq_ids.json"
    esm_path = out / "esm2_token_bins.npy"
    sap_path = out / "saprot_token_bins.npy"
    meta_path = out / "meta.json"
    lab_path = out / "labels.json"
    lpr_path = out / "labels_per_row.npy"
    for p in (seq_path, esm_path, sap_path, meta_path):
        if not p.is_file():
            raise SystemExit(f"缺少缓存文件: {p}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("esm2_repr_layers", "")) not in ("", "last_hidden_state_only"):
        raise SystemExit("当前 append 脚本仅支持单层 ESM（meta 中 esm2_repr_layers 非多层）；请改用手动全量重提特征。")

    ext.set_seed(2026)
    indep_rows = ext.load_split(args.indep_pos.resolve(), args.indep_neg.resolve())
    seq_map: Dict[str, str] = {r[0]: r[2] for r in indep_rows}
    label_map_new: Dict[str, int] = {r[0]: int(r[1]) for r in indep_rows}

    old_ids: List[str] = json.loads(seq_path.read_text(encoding="utf-8"))
    old_set: Set[str] = set(old_ids)
    to_add: List[Tuple[str, int, str]] = [(sid, lab, seq) for sid, lab, seq in indep_rows if sid not in old_set]
    if not to_add:
        print("没有新 id 需要追加（已全部存在于 seq_ids.json）。")
        return

    print("将追加样本数:", len(to_add), "（dry_run）" if args.dry_run else "")
    if args.dry_run:
        print("示例新 id:", [x[0] for x in to_add[:5]], "...")
        return

    esm_old = np.load(esm_path)
    sap_old = np.load(sap_path)
    tb = int(esm_old.shape[1])
    d_e = int(esm_old.shape[2])
    d_s = int(sap_old.shape[2])
    if tb != int(meta.get("token_bins", tb)) or d_e != int(meta.get("dim_esm2", d_e)) or d_s != int(meta.get("dim_saprot", d_s)):
        raise SystemExit("npy 形状与 meta.json 不一致，请检查缓存是否被手动改过")

    if not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for fn in ("esm2_token_bins.npy", "saprot_token_bins.npy", "seq_ids.json", "labels.json", "labels_per_row.npy"):
            src = out / fn
            if src.is_file():
                dst = out / f"{fn}.bak.{ts}"
                shutil.copy2(src, dst)
                print("backup:", dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    add_ids = [x[0] for x in to_add]
    esm_text = {sid: seq_map[sid] for sid in add_ids}
    sap_text = {sid: ext.make_saprot_input(seq_map[sid], "", args.missing_token) for sid in add_ids}

    emb_e, dim_e2 = ext.build_token_embeddings(
        add_ids,
        esm_text,
        args.esm2_model_dir.resolve(),
        args.embed_batch_size,
        args.max_length_esm2,
        tb,
        device,
    )
    emb_s, dim_s2 = ext.build_token_embeddings(
        add_ids,
        sap_text,
        args.saprot_model_dir.resolve(),
        args.embed_batch_size,
        args.max_length_saprot,
        tb,
        device,
    )
    if int(emb_e.shape[2]) != d_e or int(emb_s.shape[2]) != d_s:
        raise SystemExit(f"新提特征维数不匹配: esm {emb_e.shape} vs old {d_e}, sap {emb_s.shape} vs old {d_s}")

    esm_new = np.vstack([esm_old, emb_e.astype(np.float16)])
    sap_new = np.vstack([sap_old, emb_s.astype(np.float16)])
    new_ids = old_ids + add_ids

    labels_old: Dict[str, int] = json.loads(lab_path.read_text(encoding="utf-8"))
    for sid in add_ids:
        labels_old[sid] = int(label_map_new[sid])
    lpr_old = np.load(lpr_path)
    lpr_add = np.array([label_map_new[sid] for sid in add_ids], dtype=np.int64)
    lpr_new = np.concatenate([lpr_old, lpr_add], axis=0)

    np.save(esm_path, esm_new)
    np.save(sap_path, sap_new)
    seq_path.write_text(json.dumps(new_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    lab_path.write_text(json.dumps(labels_old, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(lpr_path, lpr_new)

    meta["n_samples"] = int(len(new_ids))
    meta["independent_append"] = {
        "n_added": len(add_ids),
        "pos_fasta": str(args.indep_pos.resolve()),
        "neg_fasta": str(args.indep_neg.resolve()),
        "missing_token_for_saprot": args.missing_token,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("done. n_total=", len(new_ids), "esm", esm_new.shape, "sap", sap_new.shape)


if __name__ == "__main__":
    main()
