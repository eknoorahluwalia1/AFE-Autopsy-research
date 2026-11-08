"""One command per experiment:  python experiments/run.py experiments/configs/sweep_adc_bits.yaml

Owner: Compute and automation. This stub expands a config into AFE settings;
training and evaluation hooks land with the ML pipeline.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml


def expand(cfg: dict) -> list[dict]:
    """Expand a sweep block into a list of concrete AFE parameter dicts."""
    sweep = cfg.get("sweep", {})
    keys = list(sweep)
    points = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        point = dict(cfg.get("afe", {}))
        for k, v in zip(keys, combo):
            point[k.split(".", 1)[1]] = v
        points.append(point)
    return points or [dict(cfg.get("afe", {}))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for point in expand(cfg):
        for seed in cfg.get("seeds", [0]):
            print(f"{cfg['name']} seed={seed} afe={point}")
            # TODO (Compute/ML): build AFEConfig, corrupt data, train/eval, log metrics.


if __name__ == "__main__":
    main()
