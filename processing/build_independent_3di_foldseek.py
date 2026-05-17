#!/usr/bin/env python3
"""
用 Foldseek 从 data/raw/pdb/<seq_id>.pdb 导出独立集 3Di 串，写入 FASTA（与 train/test 的 3Di 格式一致），
供 ensemble_stacking_plus_abc.py 的 --holdout_3di_fasta 使用。

流程（见 foldseek issue #314）：
  createdb PDB目录 -> lndb db_h db_ss_h -> convert2fasta db_ss -> FASTA

依赖：foldseek（conda/bioconda 或 PATH；见 env/environment_setup.txt）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Set


ROOT = Path(__file__).resolve().parents[1]


def _default_foldseek() -> Path:
    env = os.environ.get("FOLDSEEK_BIN")
    if env:
        return Path(env)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        p = Path(conda) / "bin" / "foldseek"
        if p.is_file():
            return p
    found = shutil.which("foldseek")
    return Path(found) if found else Path("foldseek")


# Foldseek 偶发拒识部分 PDB；Meta 的 missing_ratio 依赖「3Di 串长度」。
# 对失败 id 用与序列等长的占位符（字母 D 在真实 3Di 中常见），使 missing 特征与「全长有 3Di」一致。
FOLDMASK_3DI = "D"


def read_fasta_seqs(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    sid = ""
    parts: List[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid:
                    out[sid] = "".join(parts)
                sid = line[1:].strip().split()[0]
                parts = []
            else:
                parts.append(line)
        if sid:
            out[sid] = "".join(parts)
    return out


def fasta_ids(path: Path) -> List[str]:
    return list(read_fasta_seqs(path).keys())
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_fasta", type=Path, default=ROOT / "data" / "independent_set" / "pos_independent.fasta")
    ap.add_argument("--neg_fasta", type=Path, default=ROOT / "data" / "independent_set" / "neg_independent.fasta")
    ap.add_argument("--pdb_dir", type=Path, default=ROOT / "data" / "raw" / "pdb")
    ap.add_argument("--out_fasta", type=Path, default=ROOT / "data" / "independent_set" / "3di" / "independent_3di.fasta")
    ap.add_argument("--foldseek", type=Path, default=_default_foldseek())
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    fs = args.foldseek.resolve()
    if not fs.is_file():
        raise SystemExit(f"未找到 foldseek: {fs}")

    want: List[str] = []
    for fp in (args.pos_fasta, args.neg_fasta):
        if not fp.is_file():
            raise FileNotFoundError(fp)
        want.extend(read_fasta_seqs(fp).keys())
    # 去重保序
    seen: Set[str] = set()
    ids: List[str] = []
    for s in want:
        if s not in seen:
            seen.add(s)
            ids.append(s)

    missing_pdb: List[str] = []
    for sid in ids:
        p = args.pdb_dir / f"{sid}.pdb"
        if not p.is_file():
            missing_pdb.append(sid)
    if missing_pdb:
        raise SystemExit(f"缺少 PDB（前 10 个）: {missing_pdb[:10]} 共 {len(missing_pdb)} 条")

    if args.dry_run:
        print(f"dry_run: n_seq={len(ids)} foldseek={fs} -> {args.out_fasta}")
        return

    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ind3di_") as td:
        tdp = Path(td)
        pdb_links = tdp / "pdbs"
        pdb_links.mkdir()
        for sid in ids:
            src = (args.pdb_dir / f"{sid}.pdb").resolve()
            dst = pdb_links / f"{sid}.pdb"
            dst.symlink_to(src)

        db = tdp / "fsdb"
        raw_fa = tdp / "raw_3di.fasta"

        def run(cmd: List[str]) -> None:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stdout or "")
                sys.stderr.write(r.stderr or "")
                raise subprocess.CalledProcessError(r.returncode, cmd)

        run(
            [
                str(fs),
                "createdb",
                str(pdb_links),
                str(db),
                "--threads",
                str(int(args.threads)),
            ]
        )
        run([str(fs), "lndb", f"{db}_h", f"{db}_ss_h"])
        run([str(fs), "convert2fasta", f"{db}_ss", str(raw_fa)])

        text = raw_fa.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise SystemExit("convert2fasta 输出为空")

        if args.out_fasta.is_file():
            bak = args.out_fasta.with_name(
                args.out_fasta.name + f".bak.{time.strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(args.out_fasta, bak)
            print("备份:", bak)
        args.out_fasta.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out_fasta}  bytes={args.out_fasta.stat().st_size}")

    aa_by_id: dict[str, str] = {}
    aa_by_id.update(read_fasta_seqs(args.pos_fasta))
    aa_by_id.update(read_fasta_seqs(args.neg_fasta))

    got: Set[str] = set()
    for line in args.out_fasta.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(">"):
            got.add(line[1:].strip().split()[0])

    not_in = [s for s in ids if s not in got]
    if not_in:
        extra: List[str] = []
        for sid in not_in:
            aa = aa_by_id.get(sid, "")
            if not aa:
                raise SystemExit(f"无 AA 序列无法补 3Di 占位: {sid}")
            pad = FOLDMASK_3DI * len(aa)
            extra.append(f">{sid}\n{pad}\n")
            print(f"foldseek 未输出 {sid}，已用等长占位 3Di（len={len(pad)}）", file=sys.stderr)
        with args.out_fasta.open("a", encoding="utf-8") as w:
            w.write("".join(extra))
        print(f"追加占位 3Di: {len(not_in)} 条")

    got2: Set[str] = set()
    for line in args.out_fasta.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(">"):
            got2.add(line[1:].strip().split()[0])
    still = [s for s in ids if s not in got2]
    if still:
        raise SystemExit(f"3Di FASTA 仍缺: {still}")


if __name__ == "__main__":
    main()
