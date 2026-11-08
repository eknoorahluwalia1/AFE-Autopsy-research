# AFE Autopsy: Governing Plan

A simulation study using public data only. Budget $0. About 10 weeks. 9 students, 5 work areas.
Compute: AMD MI300X (ROCm, PyTorch).

## Question
- **Primary:** Which AFE non-idealities degrade ECG classification most, and what is the cheapest AFE design that stays within 1% of clean-signal accuracy?
- **Secondary:** For a fixed noise budget, is it better to spend on analog hardware or to fix the signal digitally afterward?

## Hypotheses
- **H1** Electrode impedance mismatch plus finite CMRR (mains pickup) hurts more than ADC bit depth.
- **H2** A high-pass cutoff at or above 0.5 Hz distorts the ST segment, so MI and STTC suffer most.
- **H3** Below some sampling rate and bit depth accuracy collapses; above it, extra hardware buys almost nothing.
- **H4** A small digital denoiser recovers most of the loss from a low-cost analog chain.

## Signal chain
Clean ECG -> electrode/skin -> mains + common-mode -> instrumentation amp -> analog high-pass -> anti-alias low-pass -> sampling -> ADC quantizer -> (optional digital denoiser) -> classifier.
Each block: own module, own unit test, can be switched off.

## Sweeps
One-at-a-time first, then a fractional factorial or random-search grid for interactions. 3-5 seeds per config. Parameter table in [README](README.md#what-we-sweep).

## Experiments
1. 1D ResNet on clean data (baseline)
2. Same model on corrupted data (robustness)
3. Model retrained per AFE config (training recovery)
4. DSP-only Pan-Tompkins baseline (is deep learning needed?)
5. Digital denoiser (H4, analog vs digital)

## Metrics
Macro AUROC (primary), macro F1, per-class AUROC, drop vs clean (pp). Cost: bits x sample rate power proxy plus a simple energy model with written assumptions. Final figure: accuracy vs hardware cost heatmap and Pareto frontier.

## Validation of the simulator
- Zero noise + 16 bits + high rate reproduces clean accuracy
- Mains tone at exactly 50/60 Hz in the PSD
- SQNR about 6.02 N + 1.76 dB
- Filter frequency responses match design (unit tests)
- Simulated vs real noise: PSD and SNR vs MIT-BIH NST

## Roles
See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Timeline
| Weeks | Milestone |
|---|---|
| 1-2 | Literature check, environment, data |
| 3-4 | Build and validate the AFE simulator |
| 5-6 | Clean baseline and DSP baseline |
| 7-8 | Sweeps and denoiser experiment |
| 9-10 | Analysis, figures, writing, mentor review |

## Honesty rules and risks
Simulated noise may not match real noise (compare to NST, report weak spots). Official PTB-XL folds only. Disproved hypotheses are findings; report all four verdicts. Check literature before claiming novelty. Check Exea Labs terms before heavy MI300X runs. No hardware purchases.
