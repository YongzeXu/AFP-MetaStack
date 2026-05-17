#!/usr/bin/env python3
"""
对「已存在于 afp_e_esm2_saprot_token_bins 缓存中的」独立集 IND* 行，仅重算 saprot_token_bins.npy，
SaProt 输入改为 make_saprot_input(AA, 3Di, missing_token)，3Di 来自独立集 FASTA（如 Foldseek 生成）。

不动 esm2_token_bins.npy / seq_ids 顺序 / labels。

前置：
  - data/independent_set/3di/independent_3di.fasta（可先跑 processing/build_independent_3di_foldseek.py）
  - 独立集已 append 进缓存（features_scripts/append_independent_token_bins.py）

完成后请重跑：
  classifier_scripts/prepare_ensemble_probs_afp_e.py --phase test ...（仅 test NPZ），
  再 ensemble_stacking_plus_abc.py，独立集 p_saprot 才会变。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _models_root() -> Path:
    import os
    return Path(os.environ.get("AFP_MODELS_ROOT", Path.home() / "models"))


def _load_extract():
    path = ROOT / "features_scripts" / "extract_esm2_saprot_token_bins.py"
    spec = importlib.util.spec_from_file_location("extract_esm2_saprot_token_bins", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 extract_esm2_saprot_token_bins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=ROOT / "features" / "afp_e_esm2_saprot_token_bins")
    ap.add_argument("--indep_pos", type=Path, default=ROOT / "data" / "independent_set" / "pos_independent.fasta")
    ap.add_argument("--indep_neg", type=Path, default=ROOT / "data" / "independent_set" / "neg_independent.fasta")
    ap.add_argument(
        "--indep_3di_fasta",
        type=Path,
        default=ROOT / "data" / "independent_set" / "3di" / "independent_3di.fasta",
    )
    ap.add_argument("--saprot_model_dir", type=Path, default=_models_root() / "westlake-repl/SaProt_650M_AF2")
    ap.add_argument("--max_length_saprot", type=int, default=2046)
    ap.add_argument("--embed_batch_size", type=int, default=4)
    ap.add_argument("--missing_token", type=str, default="#")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()

    ext = _load_extract()
    out = args.out_dir.resolve()
    seq_path = out / "seq_ids.json"
    sap_path = out / "saprot_token_bins.npy"
    meta_path = out / "meta.json"
    for p in (seq_path, sap_path, meta_path):
        if not p.is_file():
            raise SystemExit(f"缺少 {p}")

    if not args.indep_3di_fasta.is_file():
        raise SystemExit(f"缺少 3Di FASTA: {args.indep_3di_fasta}（先跑 processing/build_independent_3di_foldseek.py）")

    ext.set_seed(2026)
    di_map = ext.load_3di_map(args.indep_3di_fasta.resolve())
    rows = ext.load_split(args.indep_pos.resolve(), args.indep_neg.resolve())
    seq_map: Dict[str, str] = {r[0]: r[2] for r in rows}
    target_ids: List[str] = [r[0] for r in rows]

    old_ids: List[str] = json.loads(seq_path.read_text(encoding="utf-8"))
    row_of = {sid: i for i, sid in enumerate(old_ids)}

    missing = [sid for sid in target_ids if sid not in row_of]
    if missing:
        raise SystemExit(f"下列 id 不在缓存中（请先 append token bins）: {missing[:8]} … 共 {len(missing)}")

    sap_text: Dict[str, str] = {
        sid: ext.make_saprot_input(seq_map[sid], di_map.get(sid, ""), args.missing_token) for sid in target_ids
    }

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tb = int(meta.get("token_bins", 32))

    if args.dry_run:
        print(f"dry_run: 将重算 SaProt 行数={len(target_ids)} token_bins={tb} 3Di 文件={args.indep_3di_fasta}")
        print("示例:", target_ids[0], "sap_in_len", len(sap_text[target_ids[0]]))
        return

    if not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = out / f"saprot_token_bins.npy.bak.refresh_ind_3di.{ts}"
        shutil.copy2(sap_path, bak)
        print("backup:", bak)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_s, _dim_s = ext.build_token_embeddings(
        target_ids,
        sap_text,
        args.saprot_model_dir.resolve(),
        int(args.embed_batch_size),
        int(args.max_length_saprot),
        tb,
        device,
    )

    sap = np.load(sap_path).astype(np.float16).copy()
    if sap.shape[1] != emb_s.shape[1] or sap.shape[2] != emb_s.shape[2]:
        raise SystemExit(f"维数不匹配: cache {sap.shape} vs new {emb_s.shape}")
    for j, sid in enumerate(target_ids):
        sap[row_of[sid]] = emb_s[j].astype(np.float16)
    np.save(sap_path, sap)

    meta["independent_saprot_3di_refresh"] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_rows": len(target_ids),
        "indep_3di_fasta": str(args.indep_3di_fasta.resolve()),
        "missing_token": str(args.missing_token),
        "saprot_model_dir": str(args.saprot_model_dir.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done. wrote", sap_path, "meta updated")


if __name__ == "__main__":
    main()
