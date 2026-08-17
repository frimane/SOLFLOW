# ===== SECTION: config ======================================================
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import argparse
import glob
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# ============================================================================
# The single control surface. Every behaviour of the pipeline is set here; the
# logic below reads the config and hardcodes nothing. --config merges a YAML
# file over these defaults (deep merge, so partial overrides are fine).

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 1990,

    "paths": {
        # # glob for the raw per-station CSVs
        # "raw_glob": "downloadData/processed_surfrad/Desert_Rock_NV/*_processed.csv",
        "windows":  "data/day_ahead_windows.npz",
        "results_dir": "results/day_ahead",
        "models_dir":  "models/day_ahead",
        # optional tag folded into checkpoint filenames so runs with different
        # stations/configs written to the same models_dir don't overwrite each
        # other (e.g. "tbl_h3f1"). Empty -> legacy names flow_<rep>_<prior>_foldK.
        "run_tag": "pooled"   # keeps checkpoints separate from single-site runs
    },

    # ---- static site features (for pooled / multi-location training) --------
    # When training on data from ONE site these are optional metadata. When you
    # POOL multiple stations, set condition_site true so the network is TOLD
    # which site each window is from, via normalized lat/lon/altitude broadcast
    # as constant horizon channels (see design note in the model section). This
    # is the location information zenith cannot carry: two sites at the same
    # latitude have the same zenith but different altitude/climate, and altitude
    # in particular separates them here. Static features generalize to unseen
    # sites (just supply their coordinates), unlike a bare learned station id.
    #
    # SINGLE SITE: leave `sites` empty and fill `site` with the one location.
    # MULTI SITE: put one entry per station in `sites` (each with a name, its
    # coordinates, and a raw_glob pointing at THAT station's processed CSVs).
    # preprocess() then processes each site separately (so each keeps its own
    # clear-sky-based CSI), builds day-windows WITHIN each site (so no window
    # ever straddles two stations -- their day ordinals are offset apart), tags
    # every window with its site id + normalized coordinates, and concatenates.
    # With condition_site on, each window is conditioned on ITS OWN coordinates.
    # The *_ref/*_span normalization constants live in `site` and are SHARED by
    # all stations so their coordinates land on one comparable scale.
    "site": {
        # "latitude":  40.12498,     # Table Mountain, CO (deg N) -- single-site
        # "longitude": -105.23680,   # deg E, west negative
        # "altitude":  1689.0,       # m
        # normalization constants so the raw values enter the net at ~unit
        # scale (CONUS-ish ranges); fixed, not fit, so a checkpoint's site
        # encoding is reproducible across runs and deployable to new sites.
        # SHARED across all stations in `sites` below.
        "lat_ref": 40.0, "lat_span": 15.0,      # ~ (lat-40)/15
        "lon_ref": -95.0, "lon_span": 25.0,     # ~ (lon+95)/25
        "alt_ref": 0.0, "alt_span": 2000.0,     # ~ alt/2000
    },
    # Multi-site pool. Empty -> single-site mode using `site` + paths.raw_glob.
    # Each entry: {"name","latitude","longitude","altitude","raw_glob"}. The
    # normalization refs above are reused for every entry.
    # Example:
    #   "sites": [
    #     {"name":"Table_Mountain_CO","latitude":40.12498,"longitude":-105.2368,
    #      "altitude":1689.0,
    #      "raw_glob":"downloadData/processed_surfrad/Table_Mountain_CO/*_processed.csv"},
    #     {"name":"Desert_Rock_NV","latitude":36.62373,"longitude":-116.01947,
    #      "altitude":1007.0,
    #      "raw_glob":"downloadData/processed_surfrad/Desert_Rock_NV/*_processed.csv"},
    #   ]
    "sites" : [
        {
            "name": "Bondville_IL",
            "latitude": 40.05192,
            "longitude": -88.37309,
            "altitude": 213.0,
            "raw_glob": "downloadData/processed_surfrad/Bondville_IL/*_processed.csv"
        },
        {
            "name": "Table_Mountain_CO",
            "latitude": 40.12498,
            "longitude": -105.23680,
            "altitude": 1689.0,
            "raw_glob": "downloadData/processed_surfrad/Table_Mountain_CO/*_processed.csv"
        },
        {
            "name": "Fort_Peck_MT",
            "latitude": 48.30783,
            "longitude": -105.10170,
            "altitude": 634.0,
            "raw_glob": "downloadData/processed_surfrad/Fort_Peck_MT/*_processed.csv"
        },
        {
            "name": "Desert_Rock_NV",
            "latitude": 36.62373,
            "longitude": -116.01947,
            "altitude": 1007.0,
            "raw_glob": "downloadData/processed_surfrad/Desert_Rock_NV/*_processed.csv"
        },
        {
            "name": "Penn_State_PA",
            "latitude": 40.72012,
            "longitude": -77.93085,
            "altitude": 376.0,
            "raw_glob": "downloadData/processed_surfrad/Penn_State_PA/*_processed.csv"
        },
        {
            "name": "Goodwin_Creek_MS",
            "latitude": 34.2547,
            "longitude": -89.8729,
            "altitude": 98.0,
            "raw_glob": "downloadData/processed_surfrad/Goodwin_Creek_MS/*_processed.csv"
        },
        {
            "name": "Sioux_Falls_SD",
            "latitude": 43.73403,
            "longitude": -96.62328,
            "altitude": 473.0,
            "raw_glob": "downloadData/processed_surfrad/Sioux_Falls_SD/*_processed.csv"
        },
    ], 

    "data": {
        # ---- source columns -------------------------------------------------
        "col_datetime": "datetime",
        "col_ghi":      "ghi_measured",
        "col_ghi_cs":   "ghi_clearsky",
        "col_zenith":   "zenith",

        # ---- cadence --------------------------------------------------------
        # steps_per_day must equal 24*60/resolution_min; clean() asserts it.
        "resolution_min": 10,
        "steps_per_day":  144,          # 1440 / 10
        "source_resolution_min": 1,     # native cadence before resampling
        "resample": True,
        "resample_min_frac": 0.6,       # keep a bin if >= this fraction of its
                                        # source minutes are valid

        # ---- physical CSI bounds -------------------------------------------
        "clearsky_floor": 50.0,         # W/m^2; below this CSI is undefined
        "zenith_daylight_max": 85.0,    # a wide daylight window keeps the
                                        # morning/evening ramps in the profile;
                                        # the mask keeps short days honest
        "csi_min": 0.02,                # strictly > 0 (log transform needs it)
        "csi_max": 1.80,                # above real cloud enhancement

        # ---- coordinate centering ------------------------------------------
        # 'clearsky': CSI = 1 maps to exactly 0 in model space, so the
        # clear-sky-anchored prior's anchor is literally the zero vector.
        # 'mean': subtract the train mean instead.
        "center": "clearsky",

        # ---- day handling ---------------------------------------------------
        "drop_suspect_zero": True,      # zero CSI in daylight = sensor dropout
        # drop days with fewer valid daylight steps than this (deep-winter
        # stubs that are nearly all padding); 0 keeps everything
        "min_daylight_steps": 12,

        # ---- NWP (HRRR) exogenous conditioning ------------------------------
        # Day-ahead HRRR forecast fields. These are read DIRECTLY from the
        # processed SURFRAD CSVs (raw_glob): process_surfrad.py already merged
        # the HRRR forecast onto the SURFRAD 1-min grid and wrote the columns
        # below into each *_processed.csv. So fullCode does NOT re-load or
        # re-align any hrrr_*.csv -- it just picks up the already-aligned
        # columns and carries them through the SAME noon-centering as CSI.
        #
        # They are a future-known covariate: available at issue time for the
        # forecast day, so they only ever populate the horizon (future) side,
        # never history. The whole block is optional: if `enabled` is false, or
        # none of the named columns are present in the CSVs, preprocessing
        # proceeds without them and every downstream consumer falls back to the
        # no-NWP behaviour.
        "nwp": {
            "enabled": True,
            # NWP columns to carry, as they appear in the processed CSV. Each is
            # normalized downstream: irradiance-like columns by the clear-sky
            # GHI scale, tcdc (a percent) into [0,1]. Edit to match your merge.
            "channels": ["dswrf_inst_wm2", "tcdc_pct"],
            # also derive an NWP clear-sky index channel (dswrf / ghi_clearsky),
            # directly comparable to the target and used as the anchor mean for
            # the (B) NWP-anchored prior. ghi_clearsky is the SAME column the
            # target CSI uses, so numerator and denominator share one sky model.
            "derive_csi": True,
            "csi_source_col": "dswrf_inst_wm2",   # numerator for NWP-CSI
            "csi_min": 0.0, "csi_max": 1.8,       # clamp for the derived NWP-CSI
        },
    },

    # ---- forecasting task (units: days) -------------------------------------
    "task": {
        "history_days": 3,     # condition on this many past days
        "forecast_days": 1,    # predict this many days ahead
    },

    "split": {
        # 'blocked_cv' | 'forward' | 'holdout' | 'loso'. loso =
        # leave-one-station-out: only valid in multi-site mode; each fold holds
        # out ALL windows of one station and trains on the rest, the honest
        # test of whether coordinate conditioning GENERALIZES to an unseen site.
        "scheme": "blocked_cv",
        "n_folds": 4,
        # holdout: split at this date if set (train <= date < test), else use
        # a chronological fraction. Both variants are purged.
        "holdout": {"train_end": None, "train_frac": 0.8},
        # validation slice carved from the end of each training set for early
        # stopping; also purged against the remaining training windows
        "val_frac": 0.1,
    },

    "representation": {
        # how many quantile knots the 'gauss' transform fits (train data only)
        "gauss_knots": 1001,
        # probability clamp for the empirical CDF so its normal-score image
        # stays finite at the extremes
        "gauss_p_clip": 1e-6,
        # margin for the 'logit' transform so the bounds map to finite values
        "logit_margin": 1e-4,
    },

    "prior": {
        "kernel": "matern32",     # within-day correlation: matern12|matern32|rbf
        "jitter": 1.0e-5,
        # floor on the per-column std profile, as a fraction of the pooled
        # residual std, so never-observed edge columns stay well-conditioned
        "col_std_floor_frac": 0.10,
        # optional one-sided enhancement mixture on top of any correlated
        # prior: with probability `weight` a positive short-scale burst is
        # added, giving the source distribution mass above clear sky where the
        # data has it. weight 'auto' uses the train fraction of CSI above the
        # eval enhancement threshold; a float fixes it; 0 disables.
        "enhancement": {
            "weight": "auto",
            "length_scale": 1.0,        # shorter than the bulk: cloud edges
                                        # decorrelate fast
            "amplitude_frac": 0.5,      # burst std as a fraction of the pooled
                                        # residual std
        },
        # optional regime gating: scale the prior's noise amplitude by the
        # residual variability observed in train for the sky regime of the
        # most recent history day (clear / broken / overcast)
        "regime": {
            "enabled": True,
            "clear_mean_min": 0.90,     # last-day mean CSI at/above this and
            "clear_std_max": 0.08,      # std at/below this -> 'clear'
            "overcast_mean_max": 0.45,  # last-day mean CSI below this -> 'overcast'
            "mult_clip": [0.5, 2.0],    # clamp on the per-regime multiplier
        },
    },

    "model": {
        "hidden": 128,
        # 'auto' picks the smallest block count whose dilated receptive field
        # covers the forecast horizon; an int is used as-is but verified
        "n_blocks": "auto",
        "max_blocks": 8,
        "max_dilation": 64,
        "kernel_size": 3,
        "time_embed_dim": 64,
        "cond_embed_dim": 128,
        "dropout": 0.1,
        # per-position conditioning: one self-attention layer over the horizon
        # plus cross-attention from horizon to the encoded history sequence,
        # inserted midway through the TCN stack
        "attention": True,
        "n_heads": 4,
        # condition on the future clear-sky GHI profile (deterministic sky
        # geometry in physical units; folds in airmass and season)
        "condition_clearsky_ghi": True,
        # (A) condition on the NWP forecast channels on the horizon side. This
        # is gated BOTH by this flag AND by the window file actually carrying
        # `fut_nwp`; if either is missing, the NWP channels are simply not
        # appended and the net is built without them.
        "condition_nwp": True,
        # condition on static site features (normalized lat/lon/altitude) as
        # constant horizon channels. Turn this ON for pooled multi-location
        # training so the network can tell sites apart (zenith only carries
        # latitude). For single-site training it is optional and harmless; the
        # channels are constant so a single-site model just learns to ignore
        # them. Gated by this flag AND by the top-level `site` block present.
        "condition_site": True,
    },

    "train": {
        "epochs": 200,
        "batch_size": 256,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "n_sampling_steps": 10,
        "sampler": "heun",        # euler | midpoint | heun
        # mixed precision: 'off' (default; bit-identical to published runs) or
        # 'bf16' (CUDA-only autocast; ~1.5-2x on Ampere+, no GradScaler
        # needed). The loss is always computed in fp32.
        "amp": "off",
        "device": "auto",
        "verbose": 1,
        "early_stopping": True,
        "patience": 15,
        "min_delta": 1.0e-4,
        "es_every": 2,
        # the early-stopping monitor runs at deployment-like fidelity: fair
        # CRPS, this many members, this many ODE steps
        "es_members": 50,
        "es_sampling_steps": 20,
        "es_max_rows": 512,
        # cap on rows*members processed per sampling chunk (memory guard)
        "sample_chunk": 4096,
        "restore_best": True,
        # EMA of network weights: decay for the shadow copy that is actually
        # deployed. 0.999 is the standard default; set 0 to disable and
        # reproduce pre-EMA numerics. Averaging the SGD iterate sits closer to
        # the loss-basin center and samples 1-3% better at no inference cost.
        "ema": 0.999,
        # Post-hoc EMOS-style member calibration (see MemberCalibrator): fit a
        # per-solar-column affine map on the ensemble members by minimizing
        # fair CRPS on the validation slice, applied at prediction. Corrects the
        # small per-column bias/dispersion misfits that a flow -- unlike a
        # pinball head -- never gets a direct CRPS gradient for. Applied
        # SYMMETRICALLY to the flow and the deep-quantile baseline so the
        # comparison stays fair. Set false to disable.
        "calibrate": True,
        # History (conditioning) dropout: fraction of training windows whose
        # HISTORY is fully blanked each batch (mask set all-False), so the
        # model learns to forecast from NWP + solar geometry alone when no
        # usable history is available (cold start at a new site, sensor
        # outages, archive gaps). At day-ahead lead time history is the weakest
        # signal (fitted blend weight ~0.3), so modest rates (0.1-0.2) add
        # graceful degradation at little or no accuracy cost and act as a mild
        # regularizer. 0 disables (default). The all-padded-history case is
        # made NaN-safe by the masked-mean clamp and the cross-attention dummy-
        # key guard, so this only changes what the model SEES, never numerics.
        "history_dropout": 0.1,
    },

    "experiment": {
        "representations": ["raw", "log", "gauss"],
        # (B) add "nwp" here to run the NWP-anchored-prior experiment: the flow
        # source is anchored at the NWP's own CSI forecast instead of clear sky
        # / persistence / climatology. Requires data.nwp.enabled and a window
        # file carrying fut_nwp_csi; make_prior falls back to the clearsky
        # anchor with a warning if the NWP-CSI channel is absent.
        "priors": ["clearsky", "climatology", "persistence", "blend", "nwp"],
        "n_ensemble": 100,
        # baseline persistence ensemble uses this many most-recent history
        # days as members (capped at task.history_days)
        "peen_days": 3,
        # deep distributional baseline: same backbone + conditioning as the
        # flow, quantile head + pinball loss. The ablation that shows whether
        # the generative transport (priors, anchoring) adds value over direct
        # deep quantile regression on identical inputs. One model per fold.
        "deep_baseline": True,
    },

    "eval": {
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
        "enhancement_threshold": 1.1,   # score real enhancement, not sensor noise
        "coverage_lo": 0.1,
        "coverage_hi": 0.9,             # central 80% interval
        # Diebold-Mariano reference method (must be one of the baselines)
        "dm_reference": "ch_peen",
        # observation-error jitter (in CSI) added to the NWP-direct baseline so
        # its probabilistic scores are non-degenerate. 0 = pure point forecast
        # (CRPS reduces to MAE). A small value (~0.03-0.05) gives it a token
        # spread so coverage/CRPS are defined; it is a REFERENCE, not a tuned
        # model, so keep this small and fixed.
        "nwp_direct_spread": 0.0,
    },
}


# ============================================================================
