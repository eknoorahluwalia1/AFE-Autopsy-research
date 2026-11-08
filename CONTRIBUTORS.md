# Contributors

Nine people, five work areas, one project lead. Roles matter more than who holds them,
so swap names any time.

| Area | Lead | Support | Owns |
|---|---|---|---|
| Project lead | Eknoor | | Milestones, mentor relationship, final review, compute-terms check |
| Analog | Mist | Shaurya | `afe/electrode.py`, `afe/amp.py`, datasheet justification |
| DSP | shahahaha | Aadit | `afe/filters.py`, `afe/adc.py`, `models/dsp_baseline.py` |
| Machine learning | Infinity | Shuchay | `models/resnet1d.py`, `models/denoiser.py`, training pipeline |
| Compute and automation | Neegy | | Config system, MI300X runs, logging, data download, ROCm environment |
| Stats and writing | Aarav | | Seeds, error bars, plots, curve fitting, paper, citations |

**Handoffs:** Analog and DSP agree on one `afe/chain.py` interface. ML consumes whatever
`chain.py` outputs. Compute runs everything. Stats checks every number before it reaches the paper.
