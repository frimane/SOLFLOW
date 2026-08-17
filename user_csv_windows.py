# Provider-neutral user CSV -> checkpoint-compatible day-ahead windows.

# User-facing columns are deliberately generic. The adapter normalizes common
# provider names and converts them to the internal names used by the trained
# checkpoint: dswrf_inst_wm2, tcdc_pct, and nwp_csi.

from __future__ import annotations

import copy
import io
import os
import re
import tempfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import forecast_variants as fv


# Canonical names users should prefer. Values are expected in W/m², percent,
# degrees, and UTC timestamps unless an alias explicitly says otherwise.
USER_COLUMNS = {
    "timestamp": "UTC timestamp",
    "ghi": "measured global horizontal irradiance (W/m²)",
    "clear_sky_ghi": "clear-sky GHI (W/m²)",
    "solar_zenith": "solar zenith angle (degrees)",
    "nwp_shortwave": "NWP forecast shortwave irradiance (W/m²)",
    "nwp_cloud_cover": "NWP forecast cloud cover (percent)",
}

# Internal names are never required from users. This mapping accepts common
# SURFRAD, HRRR, ECMWF, ICON, AROME, and generic export conventions.
ALIASES = {
    "timestamp": ("timestamp", "datetime", "time", "valid_time", "valid_time_utc", "date_time"),
    "ghi": ("ghi", "ghi_measured", "global_horizontal_irradiance", "surface_downwelling_shortwave"),
    "clear_sky_ghi": ("clear_sky_ghi", "ghi_clearsky", "ghi_cs", "ghi_cs_wm2", "clearsky_ghi", "clear_sky_global_horizontal_irradiance"),
    "solar_zenith": ("solar_zenith", "zenith", "apparent_zenith", "solar_zenith_angle"),
    "nwp_shortwave": ("nwp_shortwave", "nwp_shortwave_wm2", "nwp_ghi", "nwp_dswrf", "dswrf", "dswrf_inst_wm2", "shortwave_radiation", "surface_solar_radiation_downwards", "ssrd", "ghi_forecast"),
    "nwp_cloud_cover": ("nwp_cloud_cover", "nwp_cloud_cover_pct", "cloud_cover", "cloud_fraction", "tcdc", "tcdc_pct", "total_cloud_cover", "tcc"),
    "qc_pass": ("qc_pass", "quality_ok", "valid_ghi", "ghi_valid"),
    "nwp_shortwave_unit": ("nwp_shortwave_unit", "shortwave_unit", "ssrd_unit"),
    "nwp_cloud_cover_unit": ("nwp_cloud_cover_unit", "cloud_cover_unit", "tcc_unit"),
    "latitude": ("latitude", "lat", "site_latitude", "station_latitude"),
    "longitude": ("longitude", "lon", "lng", "site_longitude", "station_longitude"),
    "elevation": ("elevation", "altitude", "alt", "site_elevation", "station_elevation"),
}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    if not raw:
        raise ValueError("The uploaded CSV is empty.")
    try:
        raw_df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"CSV could not be parsed: {exc}") from exc
    keyed = {_key(c): c for c in raw_df.columns}
    out = {}
    for target, names in ALIASES.items():
        matches = [keyed[_key(n)] for n in names if _key(n) in keyed]
        if matches:
            out[target] = raw_df[matches[0]]
    missing = [name for name in USER_COLUMNS if name not in out]
    if missing:
        detail = ", ".join(f"{name} ({USER_COLUMNS[name]})" for name in missing)
        raise ValueError("CSV is missing required provider-neutral fields: " + detail)
    df = pd.DataFrame(out)
    # Optional quality flags are retained only if supplied.
    return df


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common unit variants without changing model-facing names
    out = df.copy()
    for col in ("ghi", "clear_sky_ghi", "nwp_shortwave"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["solar_zenith"] = pd.to_numeric(out["solar_zenith"], errors="coerce")
    out["nwp_cloud_cover"] = pd.to_numeric(out["nwp_cloud_cover"], errors="coerce")

    # Explicit unit columns take precedence. Otherwise the simple contract is
    # W/m² for irradiance and percent for cloud cover.
    if "nwp_shortwave_unit" in out:
        units = out["nwp_shortwave_unit"].astype(str).str.lower()
        kwh = units.str.contains("kwh|wh", regex=True)
        out.loc[kwh, "nwp_shortwave"] *= 1000.0
    if "nwp_cloud_cover_unit" in out:
        units = out["nwp_cloud_cover_unit"].astype(str).str.lower()
        fraction = units.str.contains("fraction|0_1|0-1")
        out.loc[fraction, "nwp_cloud_cover"] *= 100.0
    # Provider exports sometimes encode cloud cover as [0,1] without a unit.
    finite_cloud = out["nwp_cloud_cover"].dropna()
    if len(finite_cloud) and float(finite_cloud.quantile(0.99)) <= 1.5:
        out["nwp_cloud_cover"] *= 100.0
    # Zenith is expected in degrees internally. Convert only when values make
    # the convention unambiguous (all finite values lie within radians range).
    finite_zen = out["solar_zenith"].dropna()
    if len(finite_zen) and float(finite_zen.abs().max()) <= np.pi + 0.05:
        out["solar_zenith"] = np.degrees(out["solar_zenith"])
    return out


def _canonical_csv_for_core(df: pd.DataFrame, path: str) -> None:
    out = pd.DataFrame({
        "datetime": df["timestamp"], "ghi": df["ghi"],
        "ghi_cs": df["clear_sky_ghi"], "zenith": df["solar_zenith"],
        "dswrf_inst_wm2": df["nwp_shortwave"],
        "tcdc_pct": df["nwp_cloud_cover"],
    })
    out.to_csv(path, index=False)


def _nwp_extra_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    shortwave = pd.to_numeric(df["nwp_shortwave"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        nwp_csi = shortwave / pd.to_numeric(df["clear_sky_ghi"], errors="coerce").to_numpy(float)
    return {
        "dswrf_inst_wm2": shortwave,
        "tcdc_pct": pd.to_numeric(df["nwp_cloud_cover"], errors="coerce").to_numpy(float),
        "nwp_csi": np.clip(nwp_csi, 0.0, 1.8),
    }


def _regrid_days_to_checkpoint(days: Dict[str, np.ndarray], target_K: int) -> Dict[str, np.ndarray]:
    # Pad/trim noon-centered daily grids to the checkpoint's frozen K
    target_K = int(target_K)
    source_K = int(np.asarray(days["K"]).item())
    if source_K == target_K:
        return days
    source_noon = int(np.asarray(days["noon_col"]).item())
    target_noon = target_K // 2
    shift = target_noon - source_noon
    out = dict(days)
    for key, value in days.items():
        arr = np.asarray(value)
        if arr.ndim != 2 or arr.shape[1] != source_K:
            continue
        fill = False if arr.dtype == bool else np.nan
        dst = np.full((arr.shape[0], target_K), fill, dtype=arr.dtype)
        src_start = max(0, -shift)
        dst_start = max(0, shift)
        width = min(source_K - src_start, target_K - dst_start)
        if width > 0:
            dst[:, dst_start:dst_start + width] = arr[:, src_start:src_start + width]
        out[key] = dst
    out["K"] = np.int64(target_K)
    out["noon_col"] = np.int64(target_noon)
    return out


def build_user_windows(raw: bytes, model_or_cfg, resolution_min: int = 10,
                       minimum_history_days: int = 3) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    df = _convert_units(_read_csv_bytes(raw))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="first")
    if df.empty:
        raise ValueError("CSV has no valid UTC timestamps.")
    if "qc_pass" in df:
        qc = df["qc_pass"].astype(str).str.lower().isin(("true", "1", "yes", "y"))
        df.loc[~qc, "ghi"] = np.nan

    if int(resolution_min) != 10:
        raise ValueError("User CSV input is fixed at exactly 10-minute resolution.")
    deltas = df["timestamp"].diff().dropna().dt.total_seconds().to_numpy()
    if len(deltas) and not np.allclose(deltas, 600.0, atol=1e-6):
        observed = sorted(set(round(float(x) / 60.0, 6) for x in deltas))
        raise ValueError(
            "User CSV must contain only regular 10-minute timestamps; "
            f"observed adjacent intervals in minutes: {observed[:8]}. "
            "Prepare or resample the file before uploading it.")

    # Validate the uploaded file itself, not the checkpoint's history length.
    # A checkpoint may legitimately use two history days while the user CSV
    # still contains the requested four-day minimum and several forecast rows.
    day_key = df["timestamp"].dt.floor("D")
    rows_per_day = df.groupby(day_key, sort=True).size()
    complete_days = int((rows_per_day == int(1440 / resolution_min)).sum())
    if complete_days < int(minimum_history_days):
        raise ValueError(
            f"CSV contains only {complete_days} complete calendar days; "
            f"at least {int(minimum_history_days)} are required "
            "(history days plus a forecast/testing day).")

    cfg = copy.deepcopy(getattr(model_or_cfg, "cfg", model_or_cfg) or {})
    cfg.setdefault("paths", {})["raw_glob"] = "__user_csv__.csv"
    cfg.setdefault("data", {}).update({
        "col_datetime": "datetime", "col_ghi": "ghi", "col_ghi_cs": "ghi_cs",
        "col_zenith": "zenith", "resolution_min": int(resolution_min),
        "source_resolution_min": int(resolution_min), "resample": False,
        "nwp": {"enabled": True, "channels": ["dswrf_inst_wm2", "tcdc_pct"],
                "derive_csi": True, "csi_source_col": "dswrf_inst_wm2"},
    })
    task = cfg.setdefault("task", {})
    history_days = int(task.get("history_days", 3))
    forecast_days = int(task.get("forecast_days", 1))
    # Use the checkpoint's own geometry. The upload minimum is deliberately
    # independent of history_days: extra complete days create additional
    # selectable windows even for checkpoints trained with shorter histories.
    task["history_days"] = history_days
    task["forecast_days"] = forecast_days

    with tempfile.TemporaryDirectory(prefix="forecast_user_csv_") as td:
        path = os.path.join(td, "__user_csv__.csv")
        _canonical_csv_for_core(df, path)
        cfg["paths"]["raw_glob"] = path
        raw_df = fv.core.load_raw(cfg)
        extras = _nwp_extra_columns(df)
        for name, values in extras.items():
            raw_df[name] = values[:len(raw_df)]
        cleaned = fv.core.clean(cfg, raw_df)

    days = fv.core.assemble_days(
        cfg, pd.to_datetime(cleaned["date"]).map(pd.Timestamp.toordinal).to_numpy(),
        cleaned["step"].to_numpy(), cleaned["csi"].to_numpy(),
        cleaned["zenith"].to_numpy(), cleaned["is_day"].to_numpy(),
        ghi_cs=cleaned["ghi_cs"].to_numpy(),
        extra_channels={name: cleaned[name].to_numpy(float) for name in extras},
    )
    target_K = getattr(model_or_cfg, "K", None)
    if target_K is not None:
        days = _regrid_days_to_checkpoint(days, int(target_K))
    W = fv.core.build_day_windows(cfg, days)
    if "fut_nwp_csi" not in W:
        raise ValueError("Internal conversion failed to create the fut_nwp_csi checkpoint channel.")
    # Pooled/site-conditioned checkpoints need one normalized [lat, lon, alt]
    # vector per window. Coordinates are optional for ordinary single-site
    # checkpoints, but become required by the app contract when site_vec exists.
    site_values = []
    if all(name in df for name in ("latitude", "longitude", "elevation")):
        lat = float(pd.to_numeric(df["latitude"], errors="coerce").dropna().iloc[0])
        lon = float(pd.to_numeric(df["longitude"], errors="coerce").dropna().iloc[0])
        alt = float(pd.to_numeric(df["elevation"], errors="coerce").dropna().iloc[0])
        if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)):
            raise ValueError("latitude, longitude, and elevation must be finite numeric values.")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("latitude must be within [-90, 90] and longitude within [-180, 180].")
        site_cfg = cfg.get("site")
        if site_cfg and all(key in site_cfg for key in ("lat_ref", "lon_ref", "alt_ref", "lat_span", "lon_span", "alt_span")):
            site_vec = fv.core._site_coord_vec(cfg, lat, lon, alt)
            W["site_coords"] = np.tile(site_vec.astype(np.float32), (W["fut_csi"].shape[0], 1))
            W["site_id"] = np.zeros(W["fut_csi"].shape[0], dtype=np.int64)
            site_values = [lat, lon, alt]

    meta = {
        "n_windows": int(W["fut_csi"].shape[0]), "history_days": history_days,
        "forecast_days": forecast_days, "resolution_min": int(resolution_min),
        "source_start": str(df["timestamp"].min()), "source_end": str(df["timestamp"].max()),
        "site_coordinates": site_values,
    }
    return W, meta


def blank_history(W: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(W)
    out["hist_csi"] = np.zeros_like(W["hist_csi"], dtype=np.float32)
    out["hist_zen"] = np.zeros_like(W["hist_zen"], dtype=np.float32)
    out["hist_mask"] = np.zeros_like(W["hist_mask"], dtype=bool)
    return out


__all__ = ["USER_COLUMNS", "build_user_windows", "blank_history"]
