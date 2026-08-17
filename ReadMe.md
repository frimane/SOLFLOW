# Conditional Flow Matching for Day-Ahead Solar Irradiance Forecasting

The core model is a conditional **FlowMatcher**, trained with Conditional Flow Matching (CFM), that generates physically interpretable ensembles of future clear-sky-index trajectories. The repository includes a Streamlit app for checkpoint evaluation, model comparison, CSV ingestion, and fine-tuning on new datasets.

<div align="center">

👉 [![Open in Streamlit — solflow.streamlit.app](https://img.shields.io/badge/Open%20in%20Streamlit-solflow.streamlit.app-FF4B4B?logo=streamlit&logoColor=white&style=flat-square)](https://solflow.streamlit.app/)

**🌍 No installation needed — generate forecasting data directly in your browser.**

💤 If asleep, click **"Yes, get this app back up!"** to wake it 
</div>

---

## Overview

Given recent history, solar geometry, forecast weather data, and site information, the model produces an ensemble of possible future GHI or CSI trajectories over the day-ahead horizon rather than a single point forecast. This makes it possible to capture cloud-driven ramps and daily energy behavior.

The repository also includes **DeepQuantile** as a direct comparator, along with classical baselines (persistence, analog-day, ..., and NWP-based methods).

## Installation

Requires Python [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

```bash
git clone https://github.com/frimane/SOLFLOW.git
cd SOLFLOW
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

## Using your own data

The app accepts CSV files.

**Requirements:**
- At least **four full days** of data at an exact **10-minute cadence**
- Columns: timestamp, GHI, clear-sky GHI, solar zenith angle, NWP irradiance/cloud data, and site coordinates (see the app ``About'' section)

The app validates your file before running and gives a clear error message if something's missing or incompatible.

## Fine-tuning on a new dataset

`finetune_flowmatcher.py` performs warm-start fine-tuning on a compatible checkpoint, without altering its underlying architecture.

```bash
python finetune_flowmatcher.py \
  --checkpoint models/needed_checkpoint.pt \
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

This project is licensed under the GNU General Public License v3.0 — see the [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) file for details.

## Disclaimer

This software is provided for research and evaluation purposes. Forecasts are model outputs and should not be treated as guaranteed for operational, financial, or grid-control decisions. Validate performance on your own data before any production use.

## Contact

**Azeddine Frimane** — [Azeddine.frimane@yahoo.com](mailto:Azeddine.frimane@yahoo.com)

