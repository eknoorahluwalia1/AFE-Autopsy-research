"""PTB-XL loading helpers for the clean ECG baseline."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


@dataclass(frozen=True)
class PTBXLData:
    """Single-lead signals and superclass labels, with official PTB-XL folds."""

    signals_v: np.ndarray
    labels: np.ndarray
    folds: np.ndarray
    record_ids: np.ndarray
    skipped_unlabeled: int


def _parse_scp_codes(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        parsed = value
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    else:
        parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected PTB-XL scp_codes to contain a dictionary, got {type(parsed)}")
    return parsed


def _unit_scale_to_volts(unit: str) -> float:
    scale = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6}
    key = unit.strip().lower()
    if key not in scale:
        raise ValueError(f"Unsupported WFDB signal unit {unit!r}; expected V, mV, or uV")
    return scale[key]


def load_ptbxl(
    data_dir: str | Path,
    lead: str = "II",
    sampling_rate: int = 500,
    duration_seconds: float = 10.0,
) -> PTBXLData:
    """Load a single lead and five diagnostic labels from PTB-XL.

    Labels follow the diagnostic superclass mapping in ``scp_statements.csv``.
    Records without one of the five target labels are skipped. Signals are
    returned in volts so they can later pass through the AFE simulator.
    """
    try:
        import pandas as pd
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to load PTB-XL data") from exc

    root = Path(data_dir).expanduser().resolve()
    database_file = root / "ptbxl_database.csv"
    statements_file = root / "scp_statements.csv"
    for path in (database_file, statements_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Could not find {path}. Download PTB-XL with `python data/download.py ptbxl`."
            )

    metadata = pd.read_csv(database_file)
    statements = pd.read_csv(statements_file)
    if "scp_code" in statements.columns:
        statements = statements.set_index("scp_code")
    for required in ("diagnostic", "diagnostic_class"):
        if required not in statements.columns:
            raise ValueError(f"{statements_file} is missing the {required!r} column")
    for required in ("ecg_id", "strat_fold", "scp_codes", "filename_hr"):
        if required not in metadata.columns:
            raise ValueError(f"{database_file} is missing the {required!r} column")

    is_diagnostic = pd.to_numeric(statements["diagnostic"], errors="coerce").fillna(0).astype(bool)
    superclass_by_code = {
        str(code): str(row["diagnostic_class"])
        for code, row in statements.loc[is_diagnostic].iterrows()
        if pd.notna(row["diagnostic_class"])
        and str(row["diagnostic_class"]) in CLASSES
    }

    num_samples = int(round(sampling_rate * duration_seconds))
    if num_samples < 1:
        raise ValueError("duration_seconds and sampling_rate must produce at least one sample")

    records: list[tuple[str, int, str, np.ndarray]] = []
    skipped_unlabeled = 0
    for row in metadata.itertuples(index=False):
        fold = int(row.strat_fold)
        if fold not in range(1, 11):
            raise ValueError(f"Unexpected PTB-XL official fold {fold} for ECG {row.ecg_id}")

        labels = np.zeros(len(CLASSES), dtype=np.float32)
        code_values = _parse_scp_codes(row.scp_codes)
        for code in code_values:
            superclass = superclass_by_code.get(str(code))
            if superclass is not None:
                labels[CLASSES.index(superclass)] = 1.0
        if not labels.any():
            skipped_unlabeled += 1
            continue
        records.append((str(row.filename_hr), fold, str(row.ecg_id), labels))

    if not records:
        raise ValueError("No labeled PTB-XL records were found")

    signals = np.empty((len(records), num_samples), dtype=np.float32)
    labels = np.empty((len(records), len(CLASSES)), dtype=np.float32)
    folds = np.empty(len(records), dtype=np.int8)
    record_ids = np.empty(len(records), dtype=np.int32)

    for index, (relative_path, fold, record_id, target) in enumerate(records):
        record_path = root / relative_path
        signal, fields = wfdb.rdsamp(str(record_path))
        actual_rate = float(fields["fs"])
        if actual_rate != sampling_rate:
            raise ValueError(
                f"Expected {sampling_rate} Hz for {record_path}, found {actual_rate} Hz. "
                "Use the matching PTB-XL filename column and sampling_rate."
            )
        names = list(fields["sig_name"])
        if lead not in names:
            raise ValueError(f"Lead {lead!r} is missing from {record_path}; available leads: {names}")
        lead_index = names.index(lead)
        units = list(fields["units"])
        if len(units) != len(names):
            raise ValueError(f"WFDB lead/unit metadata does not match in {record_path}")
        scale = _unit_scale_to_volts(units[lead_index])
        values = signal[:, lead_index].astype(np.float32, copy=False) * scale
        if len(values) >= num_samples:
            signals[index] = values[:num_samples]
        else:
            signals[index] = 0.0
            signals[index, : len(values)] = values
        labels[index] = target
        folds[index] = fold
        record_ids[index] = int(record_id)

    return PTBXLData(
        signals_v=signals,
        labels=labels,
        folds=folds,
        record_ids=record_ids,
        skipped_unlabeled=skipped_unlabeled,
    )
