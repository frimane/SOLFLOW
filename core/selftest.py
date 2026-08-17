# ===== SECTION: selftest ====================================================
# ============================================================================
# End-to-end checks of the invariants that break silently, on synthetic data.
# No real data needed; the torch checks run only if torch is installed.

def _synthetic_rows(rng, n_days=60, base_day=1000):

    dates, steps, csis, zens, gcss = [], [], [], [], []
    for dd in range(n_days):
        L = int(rng.integers(21, 49))
        noon = L // 2
        clear = (dd % 7 == 0)
        for s in range(L):
            dates.append(base_day + dd)
            steps.append(s + 10)                     # arbitrary clock offset
            zen = 20.0 + 60.0 * abs(s - noon) / max(noon, 1) \
                + 0.001 * s                          # unique argmin
            zens.append(min(zen, 84.9))
            gcss.append(900.0 * max(math.cos(math.radians(zen)), 0.05))
            if clear:
                csis.append(1.0)
            else:
                v = float(np.clip(rng.normal(0.75, 0.2), 0.05, 1.0))
                if rng.random() < 0.06:
                    v = float(rng.uniform(1.1, 1.5))
                csis.append(v)
    return (np.array(dates), np.array(steps), np.array(csis),
            np.array(zens), np.array(gcss))


def run_selftest():
    print("=== SELF-TEST ===")
    cfg = json_roundtrip(DEFAULT_CONFIG)
    cfg["task"]["history_days"] = 2
    cfg["task"]["forecast_days"] = 1
    cfg["data"]["min_daylight_steps"] = 12
    rng = np.random.default_rng(0)

    dates, steps, csis, zens, gcss = _synthetic_rows(rng)
    days = assemble_days(cfg, dates, steps, csis, zens,
                         np.ones(len(csis), bool), ghi_cs=gcss)
    K, noon_col = int(days["K"]), int(days["noon_col"])
    print(f"K={K} noon_col={noon_col} days={len(days['day_id'])}")

    # A) noon alignment: every day's minimum-zenith step is on the noon column
    for i in range(len(days["day_id"])):
        z = days["zen_day"][i]
        valid = days["mask_day"][i]
        zz = np.where(valid, z, np.inf)
        assert int(np.argmin(zz)) == noon_col, \
            f"day {i}: noon at col {int(np.argmin(zz))}, expected {noon_col}"
    print("A) all days noon-centered on the same column  OK")

    W = build_day_windows(cfg, days)
    N, H = W["fut_csi"].shape
    assert H == K and W["hist_csi"].shape[1] == 2 * K

    # B) representations: clear-sky code is 0 and encode/decode round-trips
    train_vals = W["fut_csi"][W["fut_mask"]]
    grid = np.linspace(cfg["data"]["csi_min"] + 1e-4,
                       cfg["data"]["csi_max"] - 1e-4, 400)
    for kind in REPRESENTATION_KINDS:
        rep = make_representation(cfg, kind).fit(train_vals)
        assert abs(rep.clearsky_code()) < 1e-9, f"{kind}: clear sky not at 0"
        back = rep.decode(rep.encode(grid))
        tol = 5e-3 if kind == "gauss" else 1e-4
        assert np.max(np.abs(back - grid)) < tol, \
            f"{kind}: round-trip error {np.max(np.abs(back-grid)):.2e}"
        z = rep.encode(np.array([np.nan, 1.0]))
        assert np.isnan(z[0]) and np.isfinite(z[1]), f"{kind}: NaN transparency"
    repg = make_representation(cfg, "gauss").fit(train_vals)
    zg = repg.encode(train_vals)
    assert abs(np.nanstd(zg) - 1.0) < 0.15, "gauss marginals not ~N(0,1)"
    print("B) all four representations: code(1)=0, round-trip, NaN-safe  OK")

    # C) purged splits: zero shared physical days between train and test
    for scheme in ("blocked_cv", "forward", "holdout"):
        c2 = json_roundtrip(cfg)
        c2["split"]["scheme"] = scheme
        for tr, te in make_folds(c2, W):
            assert len(tr) and len(te), f"{scheme}: empty side"
            tr_days = set()
            for i in tr:
                tr_days.update(range(int(W["first_day_ord"][i]),
                                     int(W["last_day_ord"][i]) + 1))
            te_days = set()
            for i in te:
                te_days.update(range(int(W["first_day_ord"][i]),
                                     int(W["last_day_ord"][i]) + 1))
            assert not (tr_days & te_days), f"{scheme}: day leakage!"
        tr, te = make_folds(json_roundtrip(cfg), W, quick=True)[0]
        tr2, va = carve_val(cfg, tr, W)
        va_days = set()
        for i in va:
            va_days.update(range(int(W["first_day_ord"][i]),
                                 int(W["last_day_ord"][i]) + 1))
        tr2_days = set()
        for i in tr2:
            tr2_days.update(range(int(W["first_day_ord"][i]),
                                  int(W["last_day_ord"][i]) + 1))
        assert not (tr2_days & va_days), "carve_val: day leakage!"
    print("C) blocked/forward/holdout/val splits share zero days  OK")

    # D) metrics are airtight: trashing padded truth (and padded clear-sky
    # weights) changes nothing
    M = 16
    pred = np.clip(rng.normal(0.8, 0.2, (N, M, H)), 0.02, 1.8).astype(np.float32)
    truth = W["fut_csi"].copy()
    mask = W["fut_mask"]
    gcs = W["fut_ghi_cs"].copy()
    e1 = evaluate(pred, truth, mask, cfg, K=K, n_days=1, ghi_cs=gcs)
    truth_c = truth.copy(); truth_c[~mask] = 999.0
    gcs_c = gcs.copy(); gcs_c[~mask] = 12345.0
    e2 = evaluate(pred, truth_c, mask, cfg, K=K, n_days=1, ghi_cs=gcs_c)
    for k in ["crps", "crps_ghi_weighted", "rmse", "rmse_ghi", "mae",
              "coverage_80", "enh_frac_err", "calibration_err"]:
        assert abs(e1[k] - e2[k]) < 1e-9, f"padded truth leaked into {k}"
    print("D) corrupting padded truth/weights moved no metric  OK")

    # E) priors: cross-night covariance exactly zero, samples finite, every
    # anchor kind works
    cfg2 = json_roundtrip(cfg)
    cfg2["task"]["forecast_days"] = 2
    W2 = build_day_windows(cfg2, days)
    rep = make_representation(cfg2, "raw").fit(W2["fut_csi"][W2["fut_mask"]])
    fill = float(rep.clearsky_code())
    x1 = rep.encode(W2["fut_csi"])
    hc = self_fill(rep.encode(W2["hist_csi"]), W2["hist_mask"], fill)
    n2 = W2["fut_mask"].shape[0]
    for kind in PRIOR_KINDS:
        pr = make_prior(cfg2, kind, K, 2)
        pr.fit_for(x1, hc, W2["hist_mask"], W2["fut_mask"],
                   rep.clearsky_code(), hist_csi_phys=W2["hist_csi"],
                   enh_frac_train=0.05)
        S = pr.full_covariance()
        assert np.allclose(S[:K, K:2*K], 0.0, atol=0.0), \
            f"{kind}: cross-night covariance not exactly 0"
        s = pr.sample(n2, np.random.default_rng(1), hist_rep=hc,
                      hist_mask=W2["hist_mask"], fut_mask=W2["fut_mask"],
                      hist_csi_phys=W2["hist_csi"])
        assert s.shape == (n2, 2 * K) and np.isfinite(s).all(), \
            f"{kind}: non-finite or misshaped sample"
    print("E) all five priors: exact night decorrelation, finite draws  OK")

    # F) dilations plan covers the horizon (or refuses without attention)
    dils, rf, covered = plan_dilations({"n_blocks": "auto", "max_blocks": 8,
                                        "max_dilation": 64, "kernel_size": 3},
                                       H_out=2 * K)
    assert covered, f"auto plan rf={rf} does not cover H_out={2*K}"
    try:
        plan_dilations({"n_blocks": 1, "kernel_size": 3,
                        "max_blocks": 8, "max_dilation": 64}, H_out=2 * K)
        # planning alone never raises; the builder decides. Emulate:
        d1, r1, c1 = plan_dilations({"n_blocks": 1, "kernel_size": 3,
                                     "max_blocks": 8, "max_dilation": 64},
                                    H_out=2 * K)
        assert not c1
    except AssertionError:
        raise
    print(f"F) dilation plan {dils} rf={rf} covers H_out={2*K}  OK")

    # G) flow model end to end (torch only): train one epoch on tiny settings,
    # sample, save, load, compare
    try:
        import torch  # noqa: F401
    except ImportError:
        print("G) (skipped: torch not installed)")
        print("=== ALL SELF-TESTS PASSED ===")
        return

    c3 = json_roundtrip(cfg)
    c3["model"].update({"hidden": 32, "cond_embed_dim": 32,
                        "time_embed_dim": 32, "n_heads": 2})
    c3["train"].update({"epochs": 2, "batch_size": 32, "verbose": 0,
                        "es_members": 8, "es_sampling_steps": 4,
                        "n_sampling_steps": 4, "sample_chunk": 256})
    rep = make_representation(c3, "gauss").fit(W["fut_csi"][W["fut_mask"]])
    pr = make_prior(c3, "blend", K, 1)
    fill = float(rep.clearsky_code())
    pr.fit_for(rep.encode(W["fut_csi"]),
               self_fill(rep.encode(W["hist_csi"]), W["hist_mask"], fill),
               W["hist_mask"], W["fut_mask"], rep.clearsky_code(),
               hist_csi_phys=W["hist_csi"], enh_frac_train=0.05)
    fmm = FlowMatcher(c3, 2 * K, K, K, 1, pr, rep, "cpu")
    n_tr = int(0.9 * N)
    fmm.fit(W["hist_csi"], W["fut_csi"], W["hist_zen"], W["fut_zen"],
            W["hist_mask"], W["fut_mask"], fut_ghi_cs=W["fut_ghi_cs"],
            es_split=(np.arange(n_tr), np.arange(n_tr, N)),
            rng=np.random.default_rng(0))
    pred = fmm.predict_ensemble(W["hist_csi"][:6], W["hist_zen"][:6],
                                W["fut_zen"][:6], W["hist_mask"][:6],
                                W["fut_mask"][:6],
                                fut_ghi_cs=W["fut_ghi_cs"][:6], n_ensemble=8)
    assert pred.shape == (6, 8, K) and np.isfinite(pred).all()
    assert (pred >= cfg["data"]["csi_min"] - 1e-6).all() \
        and (pred <= cfg["data"]["csi_max"] + 1e-6).all()
    path = "/tmp/day_ahead_selftest.pt"
    fmm.save(path)
    fm2 = FlowMatcher.load(path, device="cpu")
    k0 = next(iter(fmm.net.state_dict()))
    assert np.allclose(fmm.net.state_dict()[k0].numpy(),
                       fm2.net.state_dict()[k0].numpy())
    assert abs(fm2.rep.scale - fmm.rep.scale) < 1e-12
    print("G) flow train/sample/save/load with attention conditioning  OK")

    print("=== ALL SELF-TESTS PASSED ===")


def json_roundtrip(obj):
    return json.loads(json.dumps(obj))


# ============================================================================
