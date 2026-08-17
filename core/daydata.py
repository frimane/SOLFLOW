# ===== SECTION: daydata =====================================================
# ============================================================================
# Cleaning plus the noon-centered ragged-day assembly.
#
# Pipeline:
#   load_raw + clean : QC both irradiances jointly, derive CSI, flag daylight,
#                      keep an unclipped csi_raw column, clip csi to bounds.
#   assemble_days    : group daylight rows by day and place each day on a
#                      fixed-length canonical grid of K columns, centered so
#                      the minimum-zenith step of every day sits on the same
#                      column. Padding is NaN with valid_mask = False. Rows
#                      are placed by their within-day step offset from noon,
#                      so a mid-day data gap stays a gap (mask False) instead
#                      of silently shifting the afternoon left.
#
# K is derived from TRAIN days only when a train-day list is supplied, so the
# test period cannot leak its longest day into the grid geometry. Test days
# that extend past the train envelope are truncated at the edges (the low-sun
# tail) with a warning.
#
# pandas is only needed by load_raw/clean; assemble_days works on plain numpy
# arrays and is exercised directly by the self-test.

def _infer_utc_offset_hours(df: "pd.DataFrame") -> float:

    z = df["zenith"].to_numpy(float)
    valid = np.isfinite(z)
    if valid.sum() < 100:
        warnings.warn("_infer_utc_offset_hours: too few valid zenith rows "
                      "to infer offset reliably; defaulting to 0", RuntimeWarning)
        return 0.0

    thresh = np.nanpercentile(z[valid], 1)   # lowest 1% zenith ~= near solar noon
    near_noon = valid & (z <= thresh)

    tod_hours = (df["datetime"].dt.hour
                + df["datetime"].dt.minute / 60
                + df["datetime"].dt.second / 3600).to_numpy()[near_noon]

    # circular mean, since time-of-day wraps at 24h
    ang = tod_hours / 24.0 * 2 * np.pi
    mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2 * np.pi)
    noon_utc_hour = mean_ang / (2 * np.pi) * 24.0

    offset = round(12.0 - noon_utc_hour)
    offset = ((offset + 12) % 24) - 12       # wrap into [-12, +13]
    return float(offset)


def load_raw(cfg: Dict[str, Any]):
    d = cfg["data"]
    files = sorted(glob.glob(cfg["paths"]["raw_glob"]))
    if not files:
        raise FileNotFoundError(f"no CSV files match {cfg['paths']['raw_glob']}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df[d["col_datetime"]] = pd.to_datetime(df[d["col_datetime"]], utc=True)
    df = df.rename(columns={d["col_datetime"]: "datetime", d["col_ghi"]: "ghi",
                            d["col_ghi_cs"]: "ghi_cs", d["col_zenith"]: "zenith"})
    # NWP columns already merged into the processed CSV by process_surfrad.py.
    # They sit on the SAME SURFRAD grid, so they ride through selection and the
    # resample unchanged (mean over already-aligned values is a no-op). Only
    # columns that actually exist are carried; missing ones are simply skipped
    # and the pipeline stays NWP-free.
    nc = d.get("nwp") or {}
    nwp_cols = [c for c in (nc.get("channels", []) if nc.get("enabled") else [])
                if c in df.columns]
    keep = ["datetime", "ghi", "ghi_cs", "zenith"] + nwp_cols
    df = df[keep].drop_duplicates("datetime")
    df = df.sort_values("datetime").reset_index(drop=True)

    # non-physical readings knock out BOTH irradiances so the CSI ratio never
    # mixes a valid numerator with an invalid denominator or vice versa
    nonphys = (~np.isfinite(df["ghi"])) | (df["ghi"] < 0) \
              | (~np.isfinite(df["ghi_cs"])) | (df["ghi_cs"] < 0)
    df.loc[nonphys, ["ghi", "ghi_cs"]] = np.nan

    if d.get("resample", False):
        res = f"{int(d['resolution_min'])}min"
        idx = df.set_index("datetime")
        agg = idx[["ghi", "ghi_cs", "zenith"] + nwp_cols].resample(
            res, label="left", closed="left").mean()
        cnt = idx["ghi"].resample(res, label="left", closed="left").count()
        src = int(d.get("source_resolution_min", 1))
        max_samples = max(1, int(d["resolution_min"]) // src)
        min_samples = max(1, int(np.ceil(
            float(d.get("resample_min_frac", 0.6)) * max_samples)))
        agg = agg[cnt >= min_samples]
        df = agg.reset_index()

    # SURFRAD timestamps are UTC. Flooring UTC time to a calendar day puts
    # the day boundary at UTC midnight, which for a US station falls in the
    # middle of local daylight for much of the year -- around the summer
    # solstice this splits one physical sunrise-to-sunset period across two
    # "date" values, leaving a near-empty sunset fragment that assemble_days
    # then treats as its own bogus day (visible as a stray off-noon cluster
    # in inspect_days.py). Infer a whole-hour offset from the data itself
    # (via solar-noon timing) so the day boundary lands at local midnight,
    # solidly inside the night, without needing station metadata.
    
    utc_offset_hours = _infer_utc_offset_hours(df)

    local_dt = (df["datetime"] + pd.Timedelta(hours=utc_offset_hours)) \
                 .dt.tz_localize(None)

    df["date"] = local_dt.dt.normalize()
    # within-day step index on the fixed clock grid (0 .. steps_per_day-1)
    df["step"] = ((local_dt - df["date"])
                  .dt.total_seconds() // (int(d["resolution_min"]) * 60)).astype(int)
    return df


def clean(cfg: Dict[str, Any], df):
    d = cfg["data"]
    expected = 24 * 60 // int(d["resolution_min"])
    if int(d.get("steps_per_day", expected)) != expected:
        raise ValueError(f"steps_per_day inconsistent with resolution_min "
                         f"(expected {expected})")
    csi_min, csi_max = float(d["csi_min"]), float(d["csi_max"])
    if not csi_min > 0.0:
        raise ValueError("csi_min must be > 0")

    df = df.sort_values(["date", "step"]).reset_index(drop=True)
    df["is_day"] = df["zenith"] < d["zenith_daylight_max"]
    ghi = df["ghi"].to_numpy(float)
    ghics = df["ghi_cs"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        csi = np.where(ghics > d.get("clearsky_floor", 50.0), ghi / ghics, np.nan)
    df["csi_raw"] = csi
    df["is_day"] = df["is_day"] & np.isfinite(csi)
    df["csi"] = np.clip(csi, csi_min, csi_max)
    if d.get("drop_suspect_zero", True):
        suspect = df["is_day"] & np.isfinite(csi) & (csi <= 0.0)
        df["is_day"] = df["is_day"] & ~suspect
    return df


def assemble_days(cfg, dates, steps, csi, zen, is_day,
                  ghi_cs=None, train_day_ids=None,
                  extra_channels=None) -> Dict[str, np.ndarray]:


    d = cfg["data"]
    min_steps = int(d.get("min_daylight_steps", 0))

    sel = np.asarray(is_day, dtype=bool)
    dates = np.asarray(dates)[sel]
    steps = np.asarray(steps)[sel].astype(int)
    csi = np.asarray(csi, dtype=np.float64)[sel]
    zen = np.asarray(zen, dtype=np.float64)[sel]
    have_gcs = ghi_cs is not None
    if have_gcs:
        ghi_cs = np.asarray(ghi_cs, dtype=np.float64)[sel]
    # extra channels ride the identical daylight gate so their rows line up
    # one-for-one with csi/zen below
    extra = {}
    if extra_channels:
        for nm, arr in extra_channels.items():
            extra[nm] = np.asarray(arr, dtype=np.float64)[sel]

    uniq_days = np.unique(dates)
    rows_of = {dd: np.where(dates == dd)[0] for dd in uniq_days}
    kept = [dd for dd in uniq_days if len(rows_of[dd]) >= max(min_steps, 1)]
    if not kept:
        raise ValueError("assemble_days: no day meets min_daylight_steps")

    # per-day noon step = the measured step with minimum zenith, and the
    # extents (in steps) of daylight before/after it
    noon_step, ext_left, ext_right = {}, {}, {}
    for dd in kept:
        r = rows_of[dd]
        s = steps[r]
        noon = int(s[np.argmin(zen[r])])
        noon_step[dd] = noon
        ext_left[dd] = noon - int(s.min())
        ext_right[dd] = int(s.max()) - noon

    # grid geometry from TRAIN days only, if told which those are
    if train_day_ids is not None:
        pool = [dd for dd in kept if dd in set(np.asarray(train_day_ids).tolist())]
        if not pool:
            pool = kept
    else:
        pool = kept
    left = max(ext_left[dd] for dd in pool)
    right = max(ext_right[dd] for dd in pool)
    K = left + right + 1
    noon_col = left

    D = len(kept)
    csi_day = np.full((D, K), np.nan)
    zen_day = np.full((D, K), np.nan)
    gcs_day = np.full((D, K), np.nan) if have_gcs else None
    mask_day = np.zeros((D, K), dtype=bool)
    n_valid = np.zeros(D, dtype=int)
    day_id = np.array(kept)
    # one [D,K] value grid + one [D,K] presence grid per extra channel
    extra_day = {nm: np.full((D, K), np.nan) for nm in extra}
    extra_present = {nm: np.zeros((D, K), dtype=bool) for nm in extra}

    n_edge_dropped = 0
    for i, dd in enumerate(kept):
        r = rows_of[dd]
        cols = noon_col + (steps[r] - noon_step[dd])
        inside = (cols >= 0) & (cols < K)
        n_edge_dropped += int((~inside).sum())
        cc = cols[inside]
        csi_day[i, cc] = csi[r][inside]
        zen_day[i, cc] = zen[r][inside]
        if have_gcs:
            gcs_day[i, cc] = ghi_cs[r][inside]
        mask_day[i, cc] = True
        n_valid[i] = int(inside.sum())
        # extra channels use the EXACT same column map `cc`, so an NWP value
        # lands on the same solar-relative column as the CSI it forecasts.
        # presence is marked only where the source value is finite.
        for nm in extra:
            vals = extra[nm][r][inside]
            fin = np.isfinite(vals)
            extra_day[nm][i, cc] = vals
            extra_present[nm][i, cc[fin]] = True

    if n_edge_dropped:
        warnings.warn(f"assemble_days: {n_edge_dropped} low-sun edge step(s) "
                      f"fell outside the train-derived grid (K={K}) and were "
                      f"dropped. Expected only for test days longer than every "
                      f"train day.", RuntimeWarning)

    out = {"day_id": day_id, "csi_day": csi_day, "zen_day": zen_day,
           "mask_day": mask_day, "n_valid": n_valid,
           "K": K, "noon_col": noon_col}
    if have_gcs:
        out["ghi_cs_day"] = gcs_day
    for nm in extra:
        out[f"{nm}_day"] = extra_day[nm]
        # presence rides the same generic *_day slicing; final window key
        # becomes fut_<nm>_present
        out[f"{nm}_present_day"] = extra_present[nm]
    return out


# ============================================================================
