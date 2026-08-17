# ===== SECTION: windows =====================================================
# ============================================================================
# The unit is a SEQUENCE OF DAYS. A window is history_days + forecast_days
# CONSECUTIVE calendar days (a missing day breaks the window), concatenated
# per-day blocks of length K. Each window carries its own valid mask.
#
# Every window also records the ordinal of its first and last physical day.
# Those two integers are what make the splits airtight: two windows leak into
# each other exactly when their day intervals intersect, so purging is an
# interval-overlap test, not a guess about strides.

def build_day_windows(cfg, days: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

    Hd = int(cfg["task"]["history_days"])
    Fd = int(cfg["task"]["forecast_days"])
    span = Hd + Fd
    K = int(days["K"])

    order = np.argsort(days["day_id"])
    day_id = days["day_id"][order]
    csi = days["csi_day"][order]
    zen = days["zen_day"][order]
    msk = days["mask_day"][order]
    gcs = days.get("ghi_cs_day")
    have_gcs = gcs is not None
    if have_gcs:
        gcs = gcs[order]
    # discover NWP-style extra channels (name -> [D,K] array), reordered to
    # match day_id. These are future-known covariates: we emit ONLY their
    # forecast-side slice as fut_<name>, never a history slice, because NWP's
    # information is about the day being predicted (see design note in config).
    extra_names = [k[:-4] for k in days.keys()
                   if k.endswith("_day") and k not in ("csi_day", "zen_day",
                                                       "ghi_cs_day", "mask_day")]
    extra_arrs = {nm: days[f"{nm}_day"][order] for nm in extra_names}
    fut_extra = {nm: [] for nm in extra_names}
    D = len(day_id)

    HC, FC, HZ, FZ, HM, FM = [], [], [], [], [], []
    HG, FG, FDO, FIRST, LAST = [], [], [], [], []
    for i in range(0, D - span + 1):
        block = day_id[i:i + span]
        if block[-1] - block[0] != span - 1:      # a gap day breaks the window
            continue
        h = slice(i, i + Hd)
        f = slice(i + Hd, i + span)
        HC.append(csi[h].reshape(-1)); FC.append(csi[f].reshape(-1))
        HZ.append(zen[h].reshape(-1)); FZ.append(zen[f].reshape(-1))
        HM.append(msk[h].reshape(-1)); FM.append(msk[f].reshape(-1))
        if have_gcs:
            HG.append(gcs[h].reshape(-1)); FG.append(gcs[f].reshape(-1))
        for nm in extra_names:
            fut_extra[nm].append(extra_arrs[nm][f].reshape(-1))
        FDO.append(int(day_id[i + Hd]))
        FIRST.append(int(day_id[i]))
        LAST.append(int(day_id[i + span - 1]))

    if not HC:
        raise ValueError("build_day_windows produced 0 windows; check that "
                         "days are consecutive and history+forecast fits "
                         "within contiguous runs")

    out = {
        "hist_csi": np.asarray(HC, np.float32), "fut_csi": np.asarray(FC, np.float32),
        "hist_zen": np.asarray(HZ, np.float32), "fut_zen": np.asarray(FZ, np.float32),
        "hist_mask": np.asarray(HM, bool),      "fut_mask": np.asarray(FM, bool),
        "date_ord": np.asarray(FDO, np.int64),
        "first_day_ord": np.asarray(FIRST, np.int64),
        "last_day_ord": np.asarray(LAST, np.int64),
        "K": np.int64(K), "noon_col": np.int64(days["noon_col"]),
    }
    if have_gcs:
        out["hist_ghi_cs"] = np.asarray(HG, np.float32)
        out["fut_ghi_cs"] = np.asarray(FG, np.float32)
    for nm in extra_names:
        # "<name>_present" grids arrive here too (as their own extra channel);
        # store presence as bool, everything else as float32.
        arr = np.asarray(fut_extra[nm])
        if nm.endswith("_present"):
            out[f"fut_{nm}"] = arr.astype(bool)
        else:
            out[f"fut_{nm}"] = arr.astype(np.float32)
    return out


# ---- leakage-safe chronological splits --------------------------------------
# All splits work on window indices but purge by PHYSICAL DAY intervals: a
# window enters a training set only if its [first_day, last_day] interval is
# fully outside the evaluation interval. The self-test asserts zero shared
# days between train and test on every fold.

def _purge_train(cand_idx, first, last, eval_min_day, eval_max_day,
                 before_only=False):


    cand_idx = np.asarray(cand_idx)
    f = first[cand_idx]; l = last[cand_idx]
    if before_only:
        keep = l < eval_min_day
    else:
        keep = (l < eval_min_day) | (f > eval_max_day)
    return cand_idx[keep]


def blocked_cv_folds(cfg, W):
    n = int(cfg["split"]["n_folds"])
    first, last = W["first_day_ord"], W["last_day_ord"]
    order = np.argsort(W["date_ord"], kind="stable")
    blocks = np.array_split(order, n)
    folds = []
    for b in blocks:
        test = np.sort(b)
        tmin = int(first[test].min()); tmax = int(last[test].max())
        rest = np.setdiff1d(order, b, assume_unique=False)
        train = np.sort(_purge_train(rest, first, last, tmin, tmax))
        folds.append((train, test))
    return folds


def forward_chaining_folds(cfg, W):
    n = int(cfg["split"]["n_folds"])
    first, last = W["first_day_ord"], W["last_day_ord"]
    order = np.argsort(W["date_ord"], kind="stable")
    blocks = np.array_split(order, n + 1)
    folds = []
    for k in range(1, len(blocks)):
        test = np.sort(blocks[k])
        tmin = int(first[test].min())
        cand = np.concatenate(blocks[:k])
        train = np.sort(_purge_train(cand, first, last, tmin, tmin,
                                     before_only=True))
        folds.append((train, test))
    return folds


def holdout_split(cfg, W):
    first, last = W["first_day_ord"], W["last_day_ord"]
    order = np.argsort(W["date_ord"], kind="stable")
    hold = cfg["split"].get("holdout", {}) or {}
    train_end = hold.get("train_end")
    if train_end:
        import datetime as _dt
        cut_day = _dt.date.fromisoformat(str(train_end)).toordinal()
        test = order[first[order] > cut_day]
        cand = order[last[order] <= cut_day]
        if len(test) == 0 or len(cand) == 0:
            raise ValueError(f"holdout train_end={train_end} leaves an empty side")
        tmin = int(first[test].min()); tmax = int(last[test].max())
        train = np.sort(_purge_train(cand, first, last, tmin, tmax))
        return train, np.sort(test)
    frac = float(hold.get("train_frac", 0.8))
    cut = int(len(order) * frac)
    test = np.sort(order[cut:])
    tmin = int(first[test].min()); tmax = int(last[test].max())
    train = np.sort(_purge_train(order[:cut], first, last, tmin, tmax))
    return train, test


def loso_folds(cfg, W):

    if "site_id" not in W:
        raise ValueError("split scheme 'loso' requires a multi-site window "
                         "file (set cfg['sites']); no site_id present")
    sid = W["site_id"]
    order = np.argsort(W["date_ord"], kind="stable")
    folds = []
    for s in np.unique(sid):
        test = np.sort(order[sid[order] == s])
        train = np.sort(order[sid[order] != s])
        if len(test) == 0 or len(train) == 0:
            continue
        folds.append((train, test))
    if not folds:
        raise ValueError("loso produced no usable folds")
    return folds


def make_folds(cfg, W, quick=False):
    scheme = cfg["split"].get("scheme")
    if scheme == "loso":
        return loso_folds(cfg, W)     # ignores --quick: every station is a fold
    if quick or scheme == "holdout":
        return [holdout_split(cfg, W)]
    if scheme == "forward":
        return forward_chaining_folds(cfg, W)
    return blocked_cv_folds(cfg, W)


def carve_val(cfg, train_idx, W):

    frac = float(cfg["split"].get("val_frac", 0.1))
    first, last = W["first_day_ord"], W["last_day_ord"]
    order = train_idx[np.argsort(W["date_ord"][train_idx], kind="stable")]
    n_val = max(1, int(len(order) * frac))
    val = np.sort(order[-n_val:])
    vmin = int(first[val].min()); vmax = int(last[val].max())
    tr = np.sort(_purge_train(order[:-n_val], first, last, vmin, vmax))
    return tr, val


# ============================================================================
