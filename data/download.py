"""Download the public datasets. Nothing here is ever committed (see .gitignore).

Usage:
    python data/download.py ptbxl
    python data/download.py nst

Cite PTB-XL and PhysioNet exactly as their pages instruct.
Requires network access and ``pip install wfdb``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "raw"

DATASETS = {
    "ptbxl": "ptb-xl",        # main training/eval set
    "mitdb": "mitdb",         # optional second dataset
    "nst": "nstdb",           # validation only: never train on this
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=DATASETS)
    args = parser.parse_args()

    import wfdb  # imported late so tests do not need it

    dest = ROOT / args.name
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(DATASETS[args.name], dl_dir=str(dest))
    print(f"Downloaded {args.name} to {dest}")


if __name__ == "__main__":
    main()
