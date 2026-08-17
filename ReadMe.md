# Conditional Flow Matching for Day-Ahead Solar Irradiance Forecasting

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)

The core model is a conditional **FlowMatcher**, trained with Conditional Flow Matching (CFM), that generates physically interpretable ensembles of future clear-sky-index trajectories. The repository includes a Streamlit app for checkpoint evaluation, model comparison, CSV ingestion, and fine-tuning on new datasets.

---

## Overview

Given recent history, solar geometry, forecast weather data, and site information, the model produces an ensemble of possible future GHI or CSI trajectories over the day-ahead horizon rather than a single point forecast. This makes it possible to capture cloud-driven ramps and daily energy behavior.

The repository also includes **DeepQuantile** as a direct comparator, along with classical baselines (persistence, analog-day, ..., and NWP-based methods).

## Installation

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib pyyaml streamlit torch
```

## Running the app

```bash
streamlit run app.py
```

## Using your own data

The app accepts CSV files.

**Requirements:**
- At least **four full days** of data at an exact **10-minute cadence**
- Columns: timestamp, GHI, clear-sky GHI, solar zenith angle, NWP irradiance/cloud data, and site coordinates

The app validates your file before running and gives a clear error message if something's missing or incompatible.

## Fine-tuning on a new dataset

`finetune_flowmatcher.py` performs warm-start fine-tuning on a compatible checkpoint, without altering its underlying architecture.

```bash
python finetune_flowmatcher.py \
  --checkpoint models/flowmatcher_final.pt \
  --windows data/new_dataset_windows.npz \
  --output models/flowmatcher_finetuned.pt \
  --epochs 10 \
  --lr 2e-5 \
  --device auto
```

Add `--dry-run` first to validate your data before committing to a full training run.

## Citation

The associated manuscript is in preparation.

## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.

## Disclaimer

This software is provided for research and evaluation purposes. Forecasts are model outputs and should not be treated as guaranteed for operational, financial, or grid-control decisions. Validate performance on your own data before any production use.

