#!/usr/bin/env python3

# Final warm-start fine-tuning utility for the solar FlowMatcher.

# This script fine-tunes an existing checkpoint on a new,
# compatible window file. The checkpoint remains authoritative for the model
# architecture, fitted representation, prior, NWP channel order, clear-sky
# normalization, and site-conditioning width. Only optimization settings are
# changed.


# Required window arrays
# ----------------------
# The input ``.npz`` must contain the following row-aligned arrays:

#     hist_csi, fut_csi, hist_zen, fut_zen,
#     hist_mask, fut_mask, first_day_ord, last_day_ord, date_ord, K

# Optional arrays include ``fut_ghi_cs``, the future NWP arrays expected by the
# checkpoint, and ``site_coords`` for site-conditioned pooled checkpoints.

# Typical usage
# -------------
# Fine-tune an already-preprocessed window file:

#     python finetune_flowmatcher.py \\
#         --checkpoint models/flowmatcher_final.pt \\
#         --windows data/new_windows.npz \\
#         --output models/flowmatcher_finetuned.pt \\
#         --epochs 12 --lr 2e-5 --device auto

# Validate everything without changing weights:

#     python finetune_flowmatcher.py \\
#         --checkpoint models/flowmatcher_final.pt \\
#         --windows data/new_windows.npz \\
#         --output models/flowmatcher_finetuned.pt \\
#         --dry-run

# Build the window file first from a project preprocessing configuration:

#     python finetune_flowmatcher.py \\
#         --checkpoint models/flowmatcher_final.pt \\
#         --config configs/new_dataset.yaml \\
#         --windows data/new_windows.npz \\
#         --preprocess \\
#         --output models/flowmatcher_finetuned.pt

# The output checkpoint is written atomically. A JSON sidecar with the data
# contract, split sizes, and training settings is written beside it.


from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

import core

REQUIRED_ARRAYS = {
    "hist_csi",
    "fut_csi",
    "hist_zen",
    "fut_zen",
    "hist_mask",
    "fut_mask",
    "first_day_ord",
    "last_day_ord",
    "date_ord",
    "K",
}

ROW_LEVEL_METADATA = {
    "first_day_ord",
    "last_day_ord",
    "date_ord",
    "site_id",
    "site_names",
    "site_coords",
}


# ---------------------------------------------------------------------------
# CLI and reproducibility
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Warm-start fine-tune a trained FlowMatcher checkpoint on a "
            "compatible solar/NWP window dataset."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Existing day_ahead_flow_v2 FlowMatcher checkpoint (.pt).",
    )
    parser.add_argument(
        "--windows",
        required=True,
        help="Input/output .npz file containing checkpoint-compatible windows.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .pt file for the fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML preprocessing configuration; required with --preprocess.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run core.preprocess() before loading --windows.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of additional warm-start epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Warm-start learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the checkpoint batch size for this fine-tuning run.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Chronological validation fraction after the project holdout split.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Use only the earliest N windows after validation; 0 uses all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1990,
        help="Seed for Python, NumPy, Torch, split ordering, and prior draws.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. CUDA fails clearly if requested but unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the checkpoint, windows, NWP/site contract, and split only.",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    # Seed Python, NumPy, and Torch when Torch is installed
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Configuration and window loading
# ---------------------------------------------------------------------------
def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    # Return canonical defaults merged with a YAML preprocessing config
    cfg = core.json_roundtrip(core.DEFAULT_CONFIG)
    if not path:
        return cfg
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "--preprocess requires PyYAML. Install it with: pip install pyyaml"
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"preprocessing config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        update = yaml.safe_load(handle) or {}
    if not isinstance(update, dict):
        raise ValueError("preprocessing YAML must contain a mapping at its root")
    return core._deep_update(cfg, update)


def load_windows(path: str) -> Dict[str, np.ndarray]:
    # Load an NPZ window file without pickle support
    window_path = Path(path)
    if not window_path.is_file():
        raise FileNotFoundError(f"window file does not exist: {window_path}")
    with np.load(window_path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _shape(name: str, value: np.ndarray) -> str:
    return f"{name}{tuple(np.asarray(value).shape)}"


def _require_numeric(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(np.asarray(value).dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {value.dtype}")


def _validate_row_alignment(windows: Mapping[str, np.ndarray], n: int) -> None:
    # Reject row-level arrays that cannot be indexed with the windows
    for name, value in windows.items():
        arr = np.asarray(value)
        if name == "K" or arr.ndim == 0:
            continue
        if name in ROW_LEVEL_METADATA or name.startswith("hist_") or name.startswith("fut_"):
            if arr.shape[0] != n:
                raise ValueError(
                    f"Array '{name}' has first dimension {arr.shape[0]}, "
                    f"but the window count is {n}. Every row-level array must align."
                )


def validate_windows(fm: Any, windows: Mapping[str, np.ndarray], max_rows: int = 0) -> None:
    # Validate geometry, masks, metadata, finite values, and site shape
    missing = sorted(REQUIRED_ARRAYS - set(windows))
    if missing:
        raise ValueError(f"window file is missing required arrays: {missing}")

    hist_csi = np.asarray(windows["hist_csi"])
    fut_csi = np.asarray(windows["fut_csi"])
    if hist_csi.ndim != 2 or fut_csi.ndim != 2:
        raise ValueError("hist_csi and fut_csi must both be two-dimensional arrays")
    n = int(hist_csi.shape[0])
    if n < 20:
        raise ValueError(f"only {n} windows are available; at least 20 are required")
    if max_rows < 0:
        raise ValueError("--max-rows cannot be negative")
    if max_rows and max_rows < 20:
        raise ValueError("--max-rows must be 0 or at least 20")
    if max_rows and max_rows > n:
        raise ValueError(f"--max-rows={max_rows} exceeds available windows={n}")

    _validate_row_alignment(windows, n)
    for name in REQUIRED_ARRAYS - {"K", "hist_mask", "fut_mask"}:
        _require_numeric(name, np.asarray(windows[name]))

    expected = {
        "hist_csi": (n, int(fm.H_in)),
        "hist_zen": (n, int(fm.H_in)),
        "hist_mask": (n, int(fm.H_in)),
        "fut_csi": (n, int(fm.H_out)),
        "fut_zen": (n, int(fm.H_out)),
        "fut_mask": (n, int(fm.H_out)),
    }
    for name, wanted in expected.items():
        got = tuple(np.asarray(windows[name]).shape)
        if got != wanted:
            raise ValueError(
                f"{_shape(name, windows[name])} does not match the checkpoint; "
                f"expected {wanted}"
            )

    K_arr = np.asarray(windows["K"])
    if K_arr.size != 1:
        raise ValueError(f"K must be scalar, got shape {K_arr.shape}")
    K = int(K_arr.reshape(()))
    if K != int(fm.K):
        raise ValueError(f"window K={K} does not match checkpoint K={fm.K}")
    if int(fm.H_out) != K * int(fm.n_days):
        raise ValueError(
            f"checkpoint is internally inconsistent: H_out={fm.H_out}, "
            f"K={K}, n_days={fm.n_days}"
        )

    for value_name, mask_name in (
        ("hist_csi", "hist_mask"),
        ("fut_csi", "fut_mask"),
        ("hist_zen", "hist_mask"),
        ("fut_zen", "fut_mask"),
    ):
        values = np.asarray(windows[value_name], dtype=float)
        mask = np.asarray(windows[mask_name], dtype=bool)
        if np.any(~np.isfinite(values[mask])):
            raise ValueError(
                f"{value_name} contains non-finite values at valid masked positions"
            )
        if np.any(np.asarray(mask).sum(axis=1) == 0):
            raise ValueError(f"{mask_name} contains a completely empty window")

    for mask_name in ("hist_mask", "fut_mask"):
        mask = np.asarray(windows[mask_name])
        if not np.all(np.isin(mask, [0, 1, False, True])):
            raise ValueError(f"{mask_name} must contain only boolean/0-1 values")

    if "fut_ghi_cs" in windows:
        if tuple(np.asarray(windows["fut_ghi_cs"]).shape) != (n, int(fm.H_out)):
            raise ValueError("fut_ghi_cs has the wrong shape")
        if np.any(~np.isfinite(np.asarray(windows["fut_ghi_cs"])[np.asarray(windows["fut_mask"], bool)])):
            raise ValueError("fut_ghi_cs contains non-finite valid values")

    if fm.site_vec is not None:
        site = windows.get("site_coords")
        expected_site = (n, int(np.asarray(fm.site_vec).shape[0]))
        if site is None:
            raise ValueError(
                "checkpoint uses site conditioning but windows have no site_coords"
            )
        if tuple(np.asarray(site).shape) != expected_site:
            raise ValueError(
                f"site_coords has shape {np.asarray(site).shape}; "
                f"expected {expected_site}"
            )
        if not np.all(np.isfinite(np.asarray(site, dtype=float))):
            raise ValueError("site_coords contains non-finite values")


def select_rows(windows: Dict[str, np.ndarray], max_rows: int) -> Dict[str, np.ndarray]:
    # Select earliest chronological rows while preserving scalar arrays
    if not max_rows:
        return windows
    date_ord = np.asarray(windows["date_ord"])
    order = np.argsort(date_ord, kind="stable")[: int(max_rows)]
    n = len(date_ord)
    selected: Dict[str, np.ndarray] = {}
    for key, value in windows.items():
        arr = np.asarray(value)
        selected[key] = arr[order] if arr.ndim > 0 and arr.shape[0] == n else arr
    return selected


# ---------------------------------------------------------------------------
# Conditioning contracts and training
# ---------------------------------------------------------------------------
def future_nwp(windows: Mapping[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    # Return future NWP arrays in the format consumed by FlowMatcher.fit()
    excluded = {"fut_csi", "fut_zen", "fut_mask", "fut_ghi_cs"}
    result = {
        key: np.asarray(value)
        for key, value in windows.items()
        if key.startswith("fut_") and key not in excluded
    }
    return result or None


def validate_nwp_contract(fm: Any, windows: Mapping[str, np.ndarray]) -> None:
    # Ensure the new dataset has exactly the checkpoint NWP channel contract
    actual = fm._resolve_nwp_spec(future_nwp(windows) or {})
    expected = list(fm.nwp_spec)
    if actual != expected:
        raise ValueError(
            "NWP contract mismatch. The checkpoint expects "
            f"{expected}, but the new dataset provides {actual}. "
            "Use the same configured NWP columns and channel names as during training."
        )


def fit_arrays(
    fm: Any,
    windows: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> Any:
    # Fine-tune FlowMatcher using the project training implementation
    fit_idx = np.concatenate(
        [np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)]
    )
    local_train = np.arange(len(train_idx), dtype=np.int64)
    local_val = np.arange(len(train_idx), len(fit_idx), dtype=np.int64)

    def take(name: str) -> np.ndarray:
        return np.asarray(windows[name])[fit_idx]

    nwp = future_nwp(windows)
    nwp_fit = None if nwp is None else {key: value[fit_idx] for key, value in nwp.items()}
    site = windows.get("site_coords")
    site_fit = None if site is None else np.asarray(site)[fit_idx]

    fm.fit(
        take("hist_csi"),
        take("fut_csi"),
        take("hist_zen"),
        take("fut_zen"),
        take("hist_mask"),
        take("fut_mask"),
        fut_ghi_cs=(take("fut_ghi_cs") if "fut_ghi_cs" in windows else None),
        fut_nwp=nwp_fit,
        es_split=(local_train, local_val),
        rng=np.random.default_rng(int(seed)),
        site_coords=site_fit,
        warm_start=True,
    )
    return fm


# ---------------------------------------------------------------------------
# Output and resource management
# ---------------------------------------------------------------------------
def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    fm: Any,
    windows: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Path:
    # Write a reproducibility sidecar beside the checkpoint
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "output_checkpoint": str(output_path.resolve()),
        "windows": str(Path(args.windows).resolve()),
        "model_format": "day_ahead_flow_v2",
        "representation": fm.rep.state(),
        "prior": fm.prior.state(),
        "H_in": int(fm.H_in),
        "H_out": int(fm.H_out),
        "K": int(fm.K),
        "n_days": int(fm.n_days),
        "n_windows": int(np.asarray(windows["hist_csi"]).shape[0]),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "batch_size": int(fm.cfg["train"]["batch_size"]),
        "seed": int(args.seed),
        "device": str(fm.device),
        "nwp_spec": [list(item) for item in fm.nwp_spec],
        "site_conditioning": fm.site_vec is not None,
        "warm_start": True,
        "notes": (
            "Fine-tuned from the source checkpoint. The fitted representation, "
            "prior, NWP channel contract, and site-conditioning width were preserved."
        ),
    }
    sidecar = output_path.with_suffix(".json")
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return sidecar


def save_checkpoint_atomic(fm: Any, output_path: Path) -> None:
    # Save through a temporary file so an interrupted run cannot corrupt output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        if temporary.exists():
            temporary.unlink()
        fm.save(str(temporary))
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cleanup_resources() -> None:
    # Release Python and CUDA allocations after success or failure
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.epochs > 1000:
        raise ValueError("--epochs must be between 1 and 1000")
    if not np.isfinite(args.lr) or args.lr <= 0 or args.lr > 1e-2:
        raise ValueError("--lr must be finite, greater than 0, and at most 1e-2")
    if not 0.02 <= args.val_fraction <= 0.40:
        raise ValueError("--val-fraction must be between 0.02 and 0.40")
    if args.batch_size is not None and not 1 <= args.batch_size <= 4096:
        raise ValueError("--batch-size must be between 1 and 4096")
    if args.max_rows < 0:
        raise ValueError("--max-rows cannot be negative")
    if args.preprocess and not args.config:
        raise ValueError("--preprocess requires --config")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    seed_everything(args.seed)

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"output already exists: {output_path}. Use --overwrite to replace it."
        )

    device = core.get_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"loading checkpoint: {args.checkpoint}", flush=True)
    fm = core.FlowMatcher.load(args.checkpoint, device=device)
    if not isinstance(fm, core.FlowMatcher):
        raise TypeError("the supplied checkpoint did not load as a FlowMatcher")

    # Only optimization controls are changed. The loaded checkpoint remains
    # authoritative for architecture, representation, prior, and conditioning.
    fm.cfg["seed"] = int(args.seed)
    fm.cfg["split"]["val_frac"] = float(args.val_fraction)
    fm.cfg["train"]["epochs"] = int(args.epochs)
    fm.cfg["train"]["lr"] = float(args.lr)
    if args.batch_size is not None:
        fm.cfg["train"]["batch_size"] = int(args.batch_size)

    if args.preprocess:
        preprocess_cfg = load_yaml_config(args.config)
        preprocess_cfg.setdefault("paths", {})["windows"] = str(args.windows)
        print(f"preprocessing new dataset into: {args.windows}", flush=True)
        core.preprocess(preprocess_cfg)

    windows = select_rows(load_windows(args.windows), args.max_rows)
    validate_windows(fm, windows, max_rows=args.max_rows)
    validate_nwp_contract(fm, windows)

    split_cfg = core.json_roundtrip(fm.cfg)
    split_cfg["split"]["val_frac"] = float(args.val_fraction)
    train_pool, _ = core.holdout_split(split_cfg, windows)
    train_idx, val_idx = core.carve_val(split_cfg, train_pool, windows)
    if len(train_idx) < 10 or len(val_idx) < 5:
        raise ValueError(
            f"insufficient purged fine-tuning split: train={len(train_idx)}, "
            f"validation={len(val_idx)}; provide more windows or reduce --val-fraction"
        )

    print(f"validated windows: {len(windows['hist_csi'])}", flush=True)
    print(f"checkpoint geometry: H_in={fm.H_in}, H_out={fm.H_out}, K={fm.K}, n_days={fm.n_days}", flush=True)
    print(f"fine-tuning split: train={len(train_idx)}, validation={len(val_idx)}", flush=True)
    print(f"representation: {fm.rep.state().get('kind', 'unknown')}", flush=True)
    print(f"prior: {fm.prior.state().get('kind', 'unknown')}", flush=True)
    print(f"NWP channels: {fm.nwp_spec or 'none'}", flush=True)
    print(f"site conditioning: {'yes' if fm.site_vec is not None else 'no'}", flush=True)

    if args.dry_run:
        print("dry-run complete: no weights were changed", flush=True)
        return 0

    try:
        print(f"fine-tuning for {args.epochs} epoch(s)...", flush=True)
        fit_arrays(fm, windows, train_idx, val_idx, args.seed)
        save_checkpoint_atomic(fm, output_path)
        sidecar = write_metadata(output_path, args, fm, windows, train_idx, val_idx)
        print(f"saved fine-tuned checkpoint: {output_path}", flush=True)
        print(f"saved metadata: {sidecar}", flush=True)
    finally:
        cleanup_resources()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; no partial checkpoint was retained", flush=True)
        cleanup_resources()
        raise SystemExit(130)
    except Exception as exc:
        cleanup_resources()
        raise SystemExit(f"fine-tuning failed safely: {exc}") from exc
