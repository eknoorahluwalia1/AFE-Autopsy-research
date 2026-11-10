# AFE Autopsy

AFE Autopsy is a simulation study of how wearable ECG hardware imperfections affect
ECG diagnostic classification. We pass public PTB-XL recordings through a modeled
analog front end (AFE), then compare classifier performance against a clean-signal
baseline. The main goal is to find which hardware limitations matter most and how
little hardware is needed to stay close to clean-signal performance.

The current first milestone is a reproducible clean-data baseline. It uses one lead
(II), the 500 Hz PTB-XL records, and PTB-XL's official patient-level folds. The target
is multi-label classification into the five diagnostic superclasses NORM, MI, STTC,
CD, and HYP. These are diagnostic categories, not an arrhythmia-only label set.

## Get the data

Install the Python dependencies and a PyTorch build that matches your machine. For
the AMD MI300X, install the ROCm build of PyTorch as described by the official
PyTorch instructions.

```powershell
pip install -r requirements.txt
python data/download.py ptbxl
```

This downloads PTB-XL to `data/raw/ptbxl`. Cite PTB-XL and PhysioNet as their pages
instruct. Data files are ignored by Git.

## Train the clean baseline

```powershell
python train_baseline.py
```

The command reads `baseline_clean.yaml`, trains one small 1D ResNet per configured
seed, uses folds 1-8 for training and fold 9 for validation, then evaluates the
selected checkpoint on fold 10. It reports macro AUROC, macro F1, and per-class
AUROC. Checkpoints, per-seed histories, and a seed summary go under
`outputs/baseline_clean/` and are not committed.

The defaults are a starting point for the first baseline, not reported results.
The AFE sweeps, DSP baseline, and digital denoiser are still separate work.

## Project plan

See [PLAN.md](PLAN.md) for the research questions, hypotheses, planned sweeps, and
validation limits. The paper is a simulation study. Simulated-noise realism and
hardware cost estimates will need to be reported with their assumptions, and no
result from this repository alone establishes device or clinical performance.
