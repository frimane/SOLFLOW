# ===== SECTION: run =========================================================
# ============================================================================
# Preprocessing + the experiment driver. The shape of a `run`:
#
#   preprocess : processed CSVs -> cleaned rows (+ NWP columns if enabled)
#                -> noon-centered days -> windows npz. Run inspect_days.py on
#                the same config FIRST -- it audits alignment (incl. NWP) and
#                fails loudly before any GPU time is spent.
#   run        : per fold (blocked CV / forward / holdout, all day-purged):
#                  1. fit + score the baseline ladder (see SECTION baselines)
#                  2. fit + score deep_quantile (same purged fit/val split
#                     the flow cells use -- identical information, always)
#                  3. fit + score every (representation x prior) flow cell
#                  4. DM significance vs ch_peen and vs nwp_direct; joint
#                     skill vs deep_quantile
#                Each model's metrics JSON (and .pt where applicable) is
#                written to results_dir/models/ THE MOMENT it finishes, then
#                re-written when the fold's skill stats exist -- a crash
#                mid-grid loses nothing already computed.
#   aggregate  : mean over folds of EVERY numeric metric any method reported
#                (dynamic -- new metrics can never be silently dropped from
#                the table again), with __n_folds provenance per mean and
#                n_folds_scored per method so a single-fold number is never
#                mistaken for a cross-validated one.
#   --final    : after CV validates the design, train_final fits ONE
#                deployable model per (rep, prior) on ALL windows (chrono
#                val tail for early stopping, same purge discipline) and
#                saves <run_tag>_flow_<rep>_<prior>_final.pt + .json.
#                CV checkpoints estimate skill; the _final checkpoint is
#                what you deploy and plot.
# ============================================================================
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except ImportError:
        pass


def get_device(pref="auto"):
    dev = pref
    if pref == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    # one-time backend tuning: our conv shapes are fixed per run, so letting
    # cuDNN benchmark kernels is a free speedup; TF32 matmuls are safe at this
    # model scale and enabled only on CUDA.
    if str(dev).startswith("cuda"):
        try:
            import torch
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    return dev


def self_fill(x_np, mask_np, fill):
    return np.where(mask_np, np.nan_to_num(x_np, nan=fill), fill) \
        .astype(np.float32)


def nwp_channels_from_df(cfg, df):

    nc = cfg["data"].get("nwp") or {}
    if not nc.get("enabled", False):
        return None
    d = cfg["data"]
    want = list(nc.get("channels", []))
    present = [c for c in want if c in df.columns]
    if not present and not nc.get("derive_csi", False):
        return None

    out = {}
    for col in present:
        out[col] = df[col].to_numpy(np.float64)
    missing = [c for c in want if c not in df.columns]
    if missing:
        warnings.warn(f"nwp.enabled but columns {missing} absent from the "
                      f"processed CSVs; carrying only {present}", RuntimeWarning)

    if nc.get("derive_csi", False):
        src = nc.get("csi_source_col", "dswrf_inst_wm2")
        if src in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                num = df[src].to_numpy(np.float64)
                den = df["ghi_cs"].to_numpy(np.float64)
                csi = np.where(den > d.get("clearsky_floor", 50.0),
                               num / den, np.nan)
            out["nwp_csi"] = np.clip(csi, float(nc.get("csi_min", 0.0)),
                                     float(nc.get("csi_max", 1.8)))
        else:
            warnings.warn(f"nwp: derive_csi set but {src!r} absent; "
                          f"no nwp_csi channel", RuntimeWarning)
    if not out:
        return None
    n = len(df)
    cov = int(np.isfinite(next(iter(out.values()))).sum())
    print(f"nwp: channels {list(out)} read from processed CSV, "
          f"{cov}/{n} rows finite")
    return out


def _site_coord_vec(cfg, lat, lon, alt):


    s = cfg["site"]
    return np.array([
        (float(lat) - s["lat_ref"]) / s["lat_span"],
        (float(lon) - s["lon_ref"]) / s["lon_span"],
        (float(alt) - s["alt_ref"]) / s["alt_span"],
    ], np.float32)


def preprocess(cfg):

    if cfg.get("sites"):
        return _preprocess_multi(cfg)
    return _preprocess_single(cfg)


def _preprocess_single(cfg):
    df = clean(cfg, load_raw(cfg))
    day = df[df["is_day"]]

    # optional NWP conditioning channels, read straight from the processed CSV
    # columns (already merged + aligned upstream) and carried through the same
    # noon-centering as CSI via extra_channels
    nwp_rows = nwp_channels_from_df(cfg, df)
    extra_channels = None
    if nwp_rows is not None:
        day_idx = df["is_day"].to_numpy(bool)
        extra_channels = {nm: arr[day_idx] for nm, arr in nwp_rows.items()}

    days = assemble_days(
        cfg,
        day["date"].map(lambda t: t.toordinal()).to_numpy(),
        day["step"].to_numpy(),
        day["csi"].to_numpy(),
        day["zenith"].to_numpy(),
        np.ones(len(day), bool),
        ghi_cs=day["ghi_cs"].to_numpy(),
        extra_channels=extra_channels)
    W = build_day_windows(cfg, days)

    # single-site: if condition_site is on, stamp the one site's coordinates on
    # every window so run/train share the identical per-window path as multi.
    if bool(cfg["model"].get("condition_site", False)):
        s = cfg["site"]
        vec = _site_coord_vec(cfg, s["latitude"], s["longitude"], s["altitude"])
        N = W["hist_csi"].shape[0]
        W["site_coords"] = np.tile(vec, (N, 1)).astype(np.float32)
        W["site_id"] = np.zeros(N, np.int64)
        W["site_names"] = np.asarray([str(s.get("name", "site0"))])

    os.makedirs(os.path.dirname(cfg["paths"]["windows"]) or ".", exist_ok=True)
    np.savez_compressed(cfg["paths"]["windows"], **W)
    _print_windows_summary(W)
    return W


def _repad_days(days, K_new, noon_new):

    K_old = int(days["K"]); noon_old = int(days["noon_col"])
    if K_old == K_new and noon_old == noon_new:
        return days
    offset = noon_new - noon_old
    assert offset >= 0 and offset + K_old <= K_new, \
        f"bad repad offset={offset}, K_old={K_old}, K_new={K_new}"
    D = days["day_id"].shape[0]
    out = dict(days)
    out["K"] = np.int64(K_new)
    out["noon_col"] = np.int64(noon_new)
    for key, arr in days.items():
        if key in ("day_id", "K", "noon_col", "n_valid"):
            continue
        if not (isinstance(arr, np.ndarray) and arr.ndim == 2
                and arr.shape[1] == K_old):
            continue
        if arr.dtype == bool:
            new = np.zeros((D, K_new), dtype=bool)
        else:
            new = np.full((D, K_new), np.nan, dtype=np.float64)
        new[:, offset:offset + K_old] = arr
        out[key] = new
    return out


def _preprocess_multi(cfg):
    sites = cfg["sites"]
    per_site_days = []      # (scfg, days) pairs, BEFORE windowing
    names = []
    ORD_STRIDE = 10_000_000

    for si, site in enumerate(sites):
        scfg = json_roundtrip(cfg)
        scfg["paths"] = dict(cfg["paths"])
        scfg["paths"]["raw_glob"] = site["raw_glob"]
        df = clean(scfg, load_raw(scfg))
        day = df[df["is_day"]]
        if len(day) == 0:
            warnings.warn(f"site {site.get('name', si)} has no daylight rows; "
                          f"skipping", RuntimeWarning)
            continue
        nwp_rows = nwp_channels_from_df(scfg, df)
        extra_channels = None
        if nwp_rows is not None:
            di = df["is_day"].to_numpy(bool)
            extra_channels = {nm: arr[di] for nm, arr in nwp_rows.items()}
        ords = day["date"].map(lambda t: t.toordinal()).to_numpy() \
            + si * ORD_STRIDE
        days = assemble_days(
            scfg, ords, day["step"].to_numpy(), day["csi"].to_numpy(),
            day["zenith"].to_numpy(), np.ones(len(day), bool),
            ghi_cs=day["ghi_cs"].to_numpy(), extra_channels=extra_channels)
        per_site_days.append((scfg, days))
        names.append(str(site.get("name", f"site{si}")))
        print(f"  site {names[-1]}: {len(days['day_id'])} days, "
              f"K={int(days['K'])}, noon_col={int(days['noon_col'])}")

    if not per_site_days:
        raise ValueError("multi-site preprocess produced no windows for any "
                         "station; check each site's raw_glob")

    # unify onto one shared grid: widest K, and enough room on the left that
    # every station's own noon column lands inside it
    K_max = max(int(d["K"]) for _, d in per_site_days)
    noon_max = max(int(d["noon_col"]) for _, d in per_site_days)
    # also make sure the RIGHT side has room for every station's post-noon tail
    right_max = max(int(d["K"]) - int(d["noon_col"]) for _, d in per_site_days)
    K_max = max(K_max, noon_max + right_max)

    per_site_W = []
    for si, (scfg, days) in enumerate(per_site_days):
        days_p = _repad_days(days, K_max, noon_max)
        Wi = build_day_windows(scfg, days_p)
        Ni = Wi["hist_csi"].shape[0]
        site = sites[si]
        vec = _site_coord_vec(cfg, site["latitude"], site["longitude"],
                              site["altitude"])
        Wi["site_coords"] = np.tile(vec, (Ni, 1)).astype(np.float32)
        Wi["site_id"] = np.full(Ni, si, np.int64)
        per_site_W.append(Wi)
        print(f"  site {names[si]}: {Ni} windows on shared grid "
              f"K={K_max}, noon_col={noon_max}")

    keys = [k for k in per_site_W[0]
            if k not in ("K", "noon_col", "site_names")]
    W = {}
    for k in keys:
        W[k] = np.concatenate([w[k] for w in per_site_W], axis=0)
    W["K"] = np.int64(K_max)
    W["noon_col"] = np.int64(noon_max)
    W["site_names"] = np.asarray(names)

    os.makedirs(os.path.dirname(cfg["paths"]["windows"]) or ".", exist_ok=True)
    np.savez_compressed(cfg["paths"]["windows"], **W)
    print(f"pooled {len(per_site_W)} sites -> {W['hist_csi'].shape[0]} windows")
    _print_windows_summary(W)
    return W


def _print_windows_summary(W):
    nwp_keys = [k for k in W if (k.startswith("fut_") and "nwp" in k)
                or k.startswith("fut_dswrf") or k.startswith("fut_tcdc")]
    site_note = ""
    if "site_id" in W:
        ns = len(np.unique(W["site_id"]))
        site_note = f" | sites: {ns} (coords per window: {'site_coords' in W})"
    print(f"windows: {W['hist_csi'].shape} | K={int(W['K'])} "
          f"noon_col={int(W['noon_col'])} | ghi_cs carried:",
          "fut_ghi_cs" in W, "| nwp channels:", nwp_keys, site_note)


def run_experiments(cfg, quick=False):

    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    W = dict(np.load(cfg["paths"]["windows"]))
    K = int(W["K"])
    Hd = int(cfg["task"]["history_days"])
    Fd = int(cfg["task"]["forecast_days"])
    H_in, H_out = Hd * K, Fd * K
    have_gcs = "fut_ghi_cs" in W
    thr = float(cfg["eval"]["enhancement_threshold"])
    dm_lag = Hd + Fd            # window overlap horizon, in windows
    dm_ref = cfg["eval"].get("dm_reference", "ch_peen")

    # NWP horizon channels present in the window file (fut_* excluding the
    # SURFRAD ones). A per-index slice of these is handed to fit/predict as
    # `fut_nwp`; the flow model appends the enabled ones as (A) conditioning.
    nwp_keys = [k for k in W if k.startswith("fut_")
                and k not in ("fut_csi", "fut_zen", "fut_mask", "fut_ghi_cs")]

    def slice_nwp(idx):
        if not nwp_keys:
            return None
        return {k: W[k][idx] for k in nwp_keys}

    def slice_site(idx):
        # per-window site coordinates for multi-site conditioning; None in
        # single-site mode or when condition_site is off (no site_coords key)
        return W["site_coords"][idx] if "site_coords" in W else None

    folds = make_folds(cfg, W, quick=quick)
    all_results = []
    for fi, (tr, te) in enumerate(folds):
        print(f"\n===== fold {fi+1}/{len(folds)}  "
              f"(train {len(tr)} / test {len(te)} windows) =====")
        # early-stopping validation slice, purged against its own train rows
        tr_fit, va = carve_val(cfg, tr, W)

        te_sorted = te[np.argsort(W["date_ord"][te], kind="stable")]
        gcs_te = W["fut_ghi_cs"][te_sorted] if have_gcs else None
        nwp_te = slice_nwp(te_sorted)
        site_te = slice_site(te_sorted)
        fold: Dict[str, Any] = {}
        pw_crps: Dict[str, np.ndarray] = {}     # per-window CRPS for the DM test

        def score(name, pred):
            fold[name] = evaluate(pred, W["fut_csi"][te_sorted],
                                  W["fut_mask"][te_sorted], cfg,
                                  K=K, n_days=Fd, ghi_cs=gcs_te)
            pw_crps[name] = crps_per_window(pred, W["fut_csi"][te_sorted],
                                            W["fut_mask"][te_sorted])

        def save_model_result(name):

            mdir = os.path.join(cfg["paths"]["results_dir"], "models")
            os.makedirs(mdir, exist_ok=True)
            run_tag = cfg["paths"].get("run_tag", "")
            fname = (f"{run_tag}_{name}_fold{fi}.json" if run_tag
                     else f"{name}_fold{fi}.json")
            with open(os.path.join(mdir, fname), "w") as f:
                json.dump({"model": name, "fold": fi,
                           "result": fold[name]}, f, indent=2,
                          default=_json_default)

        # ---- baselines (fit on the purged training windows) -----------------
        rng = np.random.default_rng(cfg["seed"] + fi)
        baselines = {
            "day_persistence": DayPersistence(K, Fd).fit(
                W["fut_csi"][tr_fit], W["fut_mask"][tr_fit]),
            "peen": PeEn(K, Fd, int(cfg["experiment"].get("peen_days", 3))).fit(
                W["fut_csi"][tr_fit], W["fut_mask"][tr_fit]),
            "ch_peen": CHPeEn(K, Fd).fit(
                W["fut_csi"][tr_fit], W["fut_mask"][tr_fit]),
            "analog_day": AnalogDay(K, Fd).fit(
                W["hist_csi"][tr_fit], W["fut_csi"][tr_fit],
                W["hist_mask"][tr_fit], W["fut_mask"][tr_fit]),
        }
        # NWP-direct baseline: only when the windows actually carry the NWP-CSI
        # forecast. This is the reference the NWP-conditioned flow must beat to
        # justify itself; scored on the identical masked cells as every cell.
        if "fut_nwp_csi" in W:
            baselines["nwp_direct"] = NWPDirect(
                K, Fd, spread_csi=float(cfg["eval"].get("nwp_direct_spread",
                                                        0.0))).fit(
                W["fut_csi"][tr_fit], W["fut_mask"][tr_fit])
        for name, model in baselines.items():
            pred = model.predict_ensemble(
                W["hist_csi"][te_sorted], W["hist_zen"][te_sorted],
                W["fut_zen"][te_sorted], W["hist_mask"][te_sorted],
                W["fut_mask"][te_sorted],
                fut_ghi_cs=gcs_te, fut_nwp=nwp_te,
                n_ensemble=cfg["experiment"]["n_ensemble"], rng=rng)
            score(name, pred)
            save_model_result(name)

        # ---- deep distributional baseline --------------------------------------
        # Same backbone, same conditioning, quantile head + pinball loss: the
        # ablation that isolates the generative transport. Trained per fold on
        # the same purged fit/val split as the flow cells; scored, DM-tested,
        # and incrementally saved through the identical machinery.
        fit_idx = np.concatenate([tr_fit, va])
        es_split = (np.arange(len(tr_fit)), np.arange(len(tr_fit), len(fit_idx)))
        if bool(cfg["experiment"].get("deep_baseline", True)):
            print("-- deep_quantile")
            dq_train_vals = np.concatenate([
                W["hist_csi"][tr_fit][W["hist_mask"][tr_fit]],
                W["fut_csi"][tr_fit][W["fut_mask"][tr_fit]]])
            dq_rep = make_representation(cfg, "raw").fit(dq_train_vals)
            dq = DeepQuantile(cfg, H_in, H_out, K, Fd, dq_rep, device)
            dq.fit(W["hist_csi"][fit_idx], W["fut_csi"][fit_idx],
                   W["hist_zen"][fit_idx], W["fut_zen"][fit_idx],
                   W["hist_mask"][fit_idx], W["fut_mask"][fit_idx],
                   fut_ghi_cs=(W["fut_ghi_cs"][fit_idx] if have_gcs else None),
                   fut_nwp=slice_nwp(fit_idx), es_split=es_split, rng=rng,
                   n_quantiles=int(cfg["experiment"]["n_ensemble"]),
                   site_coords=slice_site(fit_idx))
            dq.fit_calibrator(W, va, rng=rng)     # EMOS-style, on the val slice
            run_tag = cfg["paths"].get("run_tag", "")
            dq_name = (f"{run_tag}_deep_quantile_fold{fi}.pt" if run_tag
                       else f"deep_quantile_fold{fi}.pt")
            dq_ckpt = os.path.join(cfg["paths"]["models_dir"], dq_name)
            dq.save(dq_ckpt)
            pred = dq.predict_ensemble(
                W["hist_csi"][te_sorted], W["hist_zen"][te_sorted],
                W["fut_zen"][te_sorted], W["hist_mask"][te_sorted],
                W["fut_mask"][te_sorted],
                fut_ghi_cs=gcs_te, fut_nwp=nwp_te, site_coords=site_te)
            score("deep_quantile", pred)
            fold["deep_quantile"]["checkpoint"] = dq_ckpt
            save_model_result("deep_quantile")

        # ---- flow cells -------------------------------------------------------
        # the fit index order is [train..., val...] so local es indices are
        # simply a range split
        enh_frac_tr = float((W["fut_csi"][tr_fit][W["fut_mask"][tr_fit]] > thr)
                            .mean())
        for rep_kind in cfg["experiment"]["representations"]:
            train_vals = np.concatenate([
                W["hist_csi"][tr_fit][W["hist_mask"][tr_fit]],
                W["fut_csi"][tr_fit][W["fut_mask"][tr_fit]]])
            rep = make_representation(cfg, rep_kind).fit(train_vals)
            fill = float(rep.clearsky_code())
            x1_tr = rep.encode(W["fut_csi"][tr_fit])
            hc_tr = self_fill(rep.encode(W["hist_csi"][tr_fit]),
                              W["hist_mask"][tr_fit], fill)
            # (B) NWP-CSI anchor for the 'nwp' prior, in model space, on the
            # TRAIN-fit rows. Built once per representation and reused across
            # prior kinds (only the 'nwp' prior consumes it).
            nwp_anchor_tr = None
            if "fut_nwp_csi" in W:
                nwp_anchor_tr = self_fill(
                    rep.encode(W["fut_nwp_csi"][tr_fit]),
                    W["fut_mask"][tr_fit], fill)
            for prior_kind in cfg["experiment"]["priors"]:
                tag = f"flow_{rep_kind}_{prior_kind}"
                print(f"-- {tag}")
                prior = make_prior(cfg, prior_kind, K, Fd)
                prior.fit_for(x1_tr, hc_tr, W["hist_mask"][tr_fit],
                              W["fut_mask"][tr_fit], rep.clearsky_code(),
                              hist_csi_phys=W["hist_csi"][tr_fit],
                              enh_frac_train=enh_frac_tr,
                              fut_nwp_anchor=nwp_anchor_tr)
                fm = FlowMatcher(cfg, H_in, H_out, K, Fd, prior, rep, device)
                fm.fit(W["hist_csi"][fit_idx], W["fut_csi"][fit_idx],
                       W["hist_zen"][fit_idx], W["fut_zen"][fit_idx],
                       W["hist_mask"][fit_idx], W["fut_mask"][fit_idx],
                       fut_ghi_cs=(W["fut_ghi_cs"][fit_idx] if have_gcs else None),
                       fut_nwp=slice_nwp(fit_idx),
                       es_split=es_split, rng=rng,
                       site_coords=slice_site(fit_idx))
                fm.fit_calibrator(W, va, rng=rng)  # EMOS-style, on val slice
                run_tag = cfg["paths"].get("run_tag", "")
                fname = (f"{run_tag}_{tag}_fold{fi}.pt" if run_tag
                         else f"{tag}_fold{fi}.pt")
                ckpt = os.path.join(cfg["paths"]["models_dir"], fname)
                fm.save(ckpt)
                pred = fm.predict_ensemble(
                    W["hist_csi"][te_sorted], W["hist_zen"][te_sorted],
                    W["fut_zen"][te_sorted], W["hist_mask"][te_sorted],
                    W["fut_mask"][te_sorted],
                    fut_ghi_cs=gcs_te, fut_nwp=nwp_te, site_coords=site_te,
                    n_ensemble=cfg["experiment"]["n_ensemble"], rng=rng)
                score(tag, pred)
                fold[tag]["checkpoint"] = ckpt
                fold[tag]["prior_fit"] = prior.describe()
                save_model_result(tag)   # .pt already on disk; metrics now too

        # ---- significance vs the reference(s) -------------------------------
        if dm_ref in pw_crps:
            for name in fold:
                if name == dm_ref:
                    continue
                stat, p = diebold_mariano(pw_crps[name], pw_crps[dm_ref], dm_lag)
                fold[name]["dm_stat_vs_ref"] = stat
                fold[name]["dm_p_vs_ref"] = p
                fold[name]["crps_skill_vs_ref"] = skill_score(
                    fold[name]["crps"], fold[dm_ref]["crps"])
        # second reference: NWP-direct. This is the test that answers "is the
        # NWP-conditioned model actually better than just trusting the NWP?".
        # crps_skill_vs_nwp > 0 with dm_p_vs_nwp < 0.05 means the flow cell
        # significantly beats the raw NWP forecast; <= 0 means it does NOT --
        # i.e. conditioning added no value (or hurt) on that fold.
        if "nwp_direct" in pw_crps:
            for name in fold:
                if name == "nwp_direct":
                    continue
                stat, p = diebold_mariano(pw_crps[name], pw_crps["nwp_direct"],
                                          dm_lag)
                fold[name]["dm_stat_vs_nwp"] = stat
                fold[name]["dm_p_vs_nwp"] = p
                fold[name]["crps_skill_vs_nwp"] = skill_score(
                    fold[name]["crps"], fold["nwp_direct"]["crps"])
        # third comparison: vs the deep_quantile baseline, on the JOINT scores
        # that a per-step quantile head structurally cannot do well. Positive
        # skill here is the flow's core justification -- it says the generative
        # transport buys coherent trajectories the quantile head cannot. CRPS
        # skill is included too so the marginal picture sits beside the joint.
        if "deep_quantile" in fold:
            dq = fold["deep_quantile"]
            for name in fold:
                if name == "deep_quantile":
                    continue
                for mk, sk in [("crps", "crps_skill_vs_deepq"),
                               ("energy_score", "es_skill_vs_deepq"),
                               ("variogram_score", "vs_skill_vs_deepq"),
                               ("daytotal_crps", "daytotal_skill_vs_deepq"),
                               ("ramp_crps", "ramp_skill_vs_deepq")]:
                    a, b = fold[name].get(mk), dq.get(mk)
                    if a is not None and b is not None and np.isfinite(a) \
                            and np.isfinite(b):
                        fold[name][sk] = skill_score(a, b)
        # re-write the per-model JSONs now that DM/skill statistics are in, so
        # the incremental files are complete, not just crash-resilient stubs
        for name in fold:
            save_model_result(name)
        all_results.append(fold)

    # ---- aggregate over folds ------------------------------------------------
    # Aggregate EVERY numeric metric any method reported, not a hand-picked
    # list -- so newly added metrics (and enh_frac_pred/true, previously
    # dropped) always appear, and a metric that is genuinely absent for a
    # method stays absent rather than surfacing as a confusing bare nan. We
    # also record how many folds actually contributed to each mean, so a
    # single-fold value is never mistaken for a cross-validated one.
    numeric_keys = set()
    for fr in all_results:
        for mth, md in fr.items():
            for k, v in md.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_keys.add(k)
    methods = list(all_results[0].keys())
    agg = {}
    for mth in methods:
        agg[mth] = {}
        for k in sorted(numeric_keys):
            vals = [fr[mth].get(k) for fr in all_results if mth in fr]
            vals = [v for v in vals if v is not None
                    and isinstance(v, (int, float)) and np.isfinite(v)]
            if vals:
                agg[mth][k] = float(np.mean(vals))
                agg[mth][f"{k}__n_folds"] = len(vals)   # provenance of the mean
        agg[mth]["n_folds_scored"] = sum(1 for fr in all_results if mth in fr)
    return {"config": cfg, "per_fold": all_results, "aggregate": agg}


def train_final(cfg):

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["train"]["device"])
    W = dict(np.load(cfg["paths"]["windows"]))
    K = int(W["K"]); Hd = int(cfg["task"]["history_days"])
    Fd = int(cfg["task"]["forecast_days"])
    H_in, H_out = Hd * K, Fd * K
    have_gcs = "fut_ghi_cs" in W
    thr = float(cfg["eval"]["enhancement_threshold"])
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)

    nwp_keys = [k for k in W if k.startswith("fut_")
                and k not in ("fut_csi", "fut_zen", "fut_mask", "fut_ghi_cs")]
    def slice_nwp(idx):
        return None if not nwp_keys else {k: W[k][idx] for k in nwp_keys}
    def slice_site(idx):
        return W["site_coords"][idx] if "site_coords" in W else None

    # chronological order, then a tail val carve with day-purge against train
    order = np.argsort(W["date_ord"], kind="stable")
    tr_all, va = carve_val(cfg, order, W)
    fit_idx = np.concatenate([tr_all, va])
    es_split = (np.arange(len(tr_all)), np.arange(len(tr_all), len(fit_idx)))
    enh_frac_tr = float((W["fut_csi"][tr_all][W["fut_mask"][tr_all]] > thr).mean())

    run_tag = cfg["paths"].get("run_tag", "")
    saved = {}
    print(f"\n=== FINAL fit on all windows "
          f"(train {len(tr_all)} / val {len(va)}) ===")
    for rep_kind in cfg["experiment"]["representations"]:
        train_vals = np.concatenate([
            W["hist_csi"][tr_all][W["hist_mask"][tr_all]],
            W["fut_csi"][tr_all][W["fut_mask"][tr_all]]])
        rep = make_representation(cfg, rep_kind).fit(train_vals)
        fill = float(rep.clearsky_code())
        x1_tr = rep.encode(W["fut_csi"][tr_all])
        hc_tr = self_fill(rep.encode(W["hist_csi"][tr_all]),
                          W["hist_mask"][tr_all], fill)
        nwp_anchor_tr = None
        if "fut_nwp_csi" in W:
            nwp_anchor_tr = self_fill(rep.encode(W["fut_nwp_csi"][tr_all]),
                                      W["fut_mask"][tr_all], fill)
        for prior_kind in cfg["experiment"]["priors"]:
            tag = f"flow_{rep_kind}_{prior_kind}"
            print(f"-- FINAL {tag}")
            prior = make_prior(cfg, prior_kind, K, Fd)
            prior.fit_for(x1_tr, hc_tr, W["hist_mask"][tr_all],
                          W["fut_mask"][tr_all], rep.clearsky_code(),
                          hist_csi_phys=W["hist_csi"][tr_all],
                          enh_frac_train=enh_frac_tr,
                          fut_nwp_anchor=nwp_anchor_tr)
            fm = FlowMatcher(cfg, H_in, H_out, K, Fd, prior, rep, device)
            fm.fit(W["hist_csi"][fit_idx], W["fut_csi"][fit_idx],
                   W["hist_zen"][fit_idx], W["fut_zen"][fit_idx],
                   W["hist_mask"][fit_idx], W["fut_mask"][fit_idx],
                   fut_ghi_cs=(W["fut_ghi_cs"][fit_idx] if have_gcs else None),
                   fut_nwp=slice_nwp(fit_idx), es_split=es_split, rng=rng,
                   site_coords=slice_site(fit_idx))
            fm.fit_calibrator(W, va, rng=rng)   # deploy-time calibration too
            fname = (f"{run_tag}_{tag}_final.pt" if run_tag
                     else f"{tag}_final.pt")
            path = os.path.join(cfg["paths"]["models_dir"], fname)
            fm.save(path)
            # companion JSON: what this deployable model is, on what it was
            # fit, and the fitted prior diagnostics -- saved the moment this
            # model finishes so an interrupted grid keeps its completed finals
            meta = {
                "tag": tag, "checkpoint": path,
                "representation": rep_kind, "prior": prior_kind,
                "n_train_windows": int(len(tr_all)),
                "n_val_windows": int(len(va)),
                "nwp_channels": [s[0] for s in fm.nwp_spec],
                "site_vec": (None if fm.site_vec is None
                             else fm.site_vec.tolist()),
                "prior_fit": prior.describe(),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(path.replace(".pt", ".json"), "w") as f:
                json.dump(meta, f, indent=2, default=_json_default)
            saved[tag] = path
            print(f"   saved {path} (+ .json)")
    return saved


# ============================================================================
