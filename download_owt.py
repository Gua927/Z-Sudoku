#!/usr/bin/env python3
"""
Download files from the Hugging Face repo under the path `text/owt`.

Defaults to downloading only checkpoint files (*.ckpt). Use --all to download
all files under text/owt.

Examples:
  - Download all checkpoints to ./weights_owt:
      python download_owt.py

  - Download everything under text/owt:
      python download_owt.py --all

  - Download to a custom directory and from a specific revision:
      python download_owt.py --local-dir ./hf_owt --revision main
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, HfApi
except Exception:
    print(
        "[ERROR] huggingface_hub 未安装或导入失败。\n"
        "请先执行: pip install -U huggingface_hub",
        file=sys.stderr,
    )
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download files from HF repo under text/owt.")
    parser.add_argument(
        "--repo-id",
        default="chen-hao-chao/mdm-prime",
        help="Hugging Face repository id (owner/name).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision/branch/tag/commit to download from.",
    )
    parser.add_argument(
        "--local-dir",
        default=str(Path.cwd() / "weights_owt"),
        help="Directory to place the downloaded files (copied from cache).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all files under text/owt (not just checkpoints).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=("自定义 allow_patterns (覆盖 --all 与默认). 例: text/owt/**/*.ckpt text/owt/**"),
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="仅列出远程文件 (text/owt 下) 不下载。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # If only listing files, do not download
    if args.list_only:
        api = HfApi()
        print(f"[INFO] 列出远程文件: repo={args.repo_id} revision={args.revision} 前缀=text/owt/")
        try:
            files = api.list_repo_files(repo_id=args.repo_id, revision=args.revision)
        except Exception as e:
            print(f"[ERROR] 访问仓库失败: {e}")
            sys.exit(1)
        owt_files = [f for f in files if f.startswith("text/owt/")]
        if not owt_files:
            print("[WARN] 没找到任何 text/owt/ 前缀文件。")
        else:
            for f in owt_files:
                print(f)
            print(f"[INFO] 共 {len(owt_files)} 个文件。")
        return

    # Determine allow patterns for download
    if args.include:
        allow_patterns = args.include
    elif args.all:
        allow_patterns = ["text/owt/**"]
    else:
        # default: only checkpoints
        allow_patterns = ["text/owt/**/*.ckpt"]

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Starting download:")
    print(f"  repo_id     : {args.repo_id}")
    print(f"  revision    : {args.revision}")
    print(f"  allow       : {allow_patterns}")
    print(f"  local_dir   : {local_dir}")

    # Download; copy actual files (no symlinks) into local_dir
    try:
        snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            allow_patterns=allow_patterns,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        sys.exit(1)

    print(f"[INFO] Snapshot materialized at: {snapshot_path}")

    # Walk and summarize
    total_files = 0
    total_bytes = 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            total_files += 1
            fp = Path(root) / f
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                pass

    def human(n: float) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while n >= 1024 and i < len(units) - 1:
            n /= 1024
            i += 1
        return f"{n:.2f} {units[i]}"

    print(f"[INFO] Downloaded files: {total_files}")
    print(f"[INFO] Total size       : {human(total_bytes)}")
    print("[DONE]")


if __name__ == "__main__":
    main()
