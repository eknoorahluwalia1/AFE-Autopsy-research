"""Train and evaluate the clean-data PTB-XL baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import yaml

try:
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "Training needs PyTorch and scikit-learn. Install requirements.txt and a PyTorch "
        "build that matches your hardware."
    ) from exc

from data.ptbxl import CLASSES as PTBXL_CLASSES, PTBXLData, load_ptbxl
from models.resnet1d import build_resnet1d


class ECGDataset(Dataset):
    def __init__(self, data: PTBXLData, indices: np.ndarray):
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        # Keep the stored signal in volts for the AFE chain; the classifier uses mV.
        signal = torch.from_numpy(self.data.signals_v[index]).unsqueeze(0) * 1_000.0
        target = torch.from_numpy(self.data.labels[index])
        return signal, target


def _device_for(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    class_auroc: dict[str, float | None] = {}
    for index, name in enumerate(PTBXL_CLASSES):
        if np.unique(targets[:, index]).size < 2:
            class_auroc[name] = None
        else:
            class_auroc[name] = float(roc_auc_score(targets[:, index], probabilities[:, index]))
    valid_auc = [score for score in class_auroc.values() if score is not None]
    return {
        "macro_auroc": float(mean(valid_auc)) if valid_auc else None,
        "macro_f1": float(
            f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)
        ),
        "per_class_auroc": class_auroc,
    }


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for signals, targets in loader:
            logits = model(signals.to(device))
            all_targets.append(targets.numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_probabilities)


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _train_seed(
    data: PTBXLData,
    seed: int,
    training: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    lead: str,
    sampling_rate: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    train_indices = np.flatnonzero(np.isin(data.folds, np.arange(1, 9)))
    validation_indices = np.flatnonzero(data.folds == 9)
    test_indices = np.flatnonzero(data.folds == 10)
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise ValueError("PTB-XL folds 1-8, 9, and 10 must all contain labeled records")

    batch_size = int(training.get("batch_size", 32))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ECGDataset(data, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ECGDataset(data, validation_indices), batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        ECGDataset(data, test_indices), batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = build_resnet1d(in_leads=1, n_classes=len(PTBXL_CLASSES)).to(device)
    train_labels = data.labels[train_indices]
    positive = train_labels.sum(axis=0)
    negative = len(train_labels) - positive
    pos_weight = torch.as_tensor(negative / np.maximum(positive, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )

    epochs = int(training.get("epochs", 20))
    patience = int(training.get("patience", 5))
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        example_count = 0
        for signals, targets in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            example_count += len(targets)

        validation_targets, validation_probabilities = _predict(model, validation_loader, device)
        validation_metrics = _metrics(validation_targets, validation_probabilities)
        score = validation_metrics["macro_auroc"]
        if score is None:
            score = validation_metrics["macro_f1"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(example_count, 1),
                "validation_macro_auroc": (
                    float(validation_metrics["macro_auroc"])
                    if validation_metrics["macro_auroc"] is not None
                    else float("nan")
                ),
                "validation_macro_f1": float(validation_metrics["macro_f1"]),
            }
        )
        print(
            f"seed={seed} epoch={epoch}/{epochs} "
            f"loss={history[-1]['train_loss']:.4f} "
            f"val_auroc={validation_metrics['macro_auroc']} "
            f"val_f1={validation_metrics['macro_f1']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_targets, test_probabilities = _predict(model, test_loader, device)
    test_metrics = _metrics(test_targets, test_probabilities)

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    _write_history(seed_dir / "history.csv", history)
    torch.save(
        {
            "model_state_dict": best_state,
            "classes": PTBXL_CLASSES,
            "lead": lead,
            "sampling_rate": sampling_rate,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        seed_dir / "best.pt",
    )
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_auroc": (
            float(best_score) if math.isfinite(float(best_score)) else None
        ),
        "test": test_metrics,
    }
    (seed_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": results, "test_summary": {}}
    for metric in ("macro_auroc", "macro_f1"):
        values = [run["test"][metric] for run in results if run["test"][metric] is not None]
        summary["test_summary"][metric] = {
            "mean": float(mean(values)) if values else None,
            "std": float(pstdev(values)) if len(values) > 1 else (0.0 if values else None),
        }
    return summary


def run(config_path: Path, data_dir_override: Path | None, output_override: Path | None) -> None:
    config_path = config_path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config.get("data", {})
    data_dir = data_dir_override or Path(data_config.get("root", "data/raw/ptbxl"))
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    output_dir = output_override or Path(config.get("output_dir", "outputs/baseline_clean"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    sampling_rate = int(data_config.get("sampling_rate", 500))
    lead = str(data_config.get("lead", "II"))
    duration = float(data_config.get("duration_seconds", 10))
    print(f"Loading PTB-XL: {data_dir} ({sampling_rate} Hz, lead {lead})")
    data = load_ptbxl(data_dir, lead, sampling_rate, duration)
    print(
        f"Loaded {len(data.signals_v)} labeled records; skipped "
        f"{data.skipped_unlabeled} without a target superclass."
    )
    for index, name in enumerate(PTBXL_CLASSES):
        print(f"  {name}: {int(data.labels[:, index].sum())} positive records")

    device = _device_for(str(config.get("device", "auto")))
    print(f"Training device: {device}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    results = [
        _train_seed(
            data,
            int(seed),
            config.get("training", {}),
            output_dir,
            device,
            lead,
            sampling_rate,
        )
        for seed in config.get("seeds", [0])
    ]
    summary = _summarize(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    for metric, values in summary["test_summary"].items():
        print(f"test {metric}: mean={values['mean']} std={values['std']}")
    print(f"Saved baseline artifacts under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().with_name("baseline_clean.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, help="Override the PTB-XL dataset directory")
    parser.add_argument("--output-dir", type=Path, help="Override the artifact directory")
    args = parser.parse_args()
    run(args.config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
