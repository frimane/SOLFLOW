# ===== SECTION: evaluate ====================================================
# ============================================================================
# WHY THIS SECTION HAS SO MANY METRICS
# ------------------------------------
# The central empirical finding of this project shaped the metric set. A deep
# quantile baseline (same backbone, same conditioning, direct pinball loss --
# see DeepQuantile) TIES OR SLIGHTLY BEATS the flow on per-step CRPS, because
# per-step CRPS scores only the MARGINAL at each timestep, which is exactly and
# only what a quantile head optimizes. Judging a generative trajectory model by
# per-step CRPS therefore hides its entire reason to exist. So the metrics are
# deliberately layered to separate "is each timestep's marginal right" from "is
# the whole day's JOINT shape right" -- the latter being what a coherent
# ensemble can do and independent per-step quantiles structurally cannot.
#
# Every reduction is mask-aware; padded positions never enter any score. The
# self-test enforces this by trashing padded truth and asserting nothing moves.
#
# GROUP 1 -- MARGINAL / CSI-space (diagnostic, equal weight per valid step)
#   crps              Fair-CRPS averaged over valid steps. The standard
#                     probabilistic accuracy score; what the quantile head
#                     optimizes, so the fairest head-to-head on marginals.
#   rmse, mae         Point error of the ensemble mean / median. Cheap sanity;
#                     not the target (a point score can't see spread).
#   calibration_err   Mean |PIT - uniform|. Are the predicted quantiles honest?
#                     THE key companion to the variogram score below: variogram
#                     is mean-invariant and cannot see calibration, so a joint
#                     claim ("flow has better dependence structure") is only
#                     airtight when paired with "flow is also calibrated."
#   coverage_80       Empirical coverage of the central 80% interval. Should sit
#                     near 0.80; far above = intervals too wide, below = too
#                     narrow. Distinguishes over- from under-dispersion.
#   enh_frac_pred/true/err  Predicted vs observed frequency of cloud
#                     enhancement (CSI>threshold). NWP smooths the tail away;
#                     the flow can over-egg it. This is the flow's known
#                     weakness, tracked explicitly so a rep/skew change can be
#                     judged by whether err shrinks.
#
# GROUP 2 -- GHI-WEIGHTED (what the errors cost physically)
#   crps_ghi_weighted A CSI miss at a 5-degree sun is worth almost nothing in
#   rmse_ghi          energy; weighting by clear-sky GHI says what the errors
#                     cost in W/m^2, not in dimensionless CSI. Prevents a model
#                     from winning by nailing worthless dawn/dusk steps.
#
# GROUP 3 -- JOINT / MULTIVARIATE (the flow's home turf; per forecast day)
#   energy_score      Multivariate CRPS (Gneiting & Raftery 2007): ||ensemble -
#                     truth|| as whole vectors. Rewards getting the JOINT
#                     trajectory right, not each step alone. Known to have only
#                     LIMITED sensitivity to correlation errors (Pinson & Tastu
#                     2013) -- hence it is reported WITH, not instead of, the
#                     variogram score.
#   variogram_score   Variogram score order 0.5 (Scheuerer & Hamill 2015):
#                     matches predicted vs observed pairwise |y_i - y_j|. This
#                     is the metric that DIRECTLY targets temporal dependence
#                     structure -- the one a per-step quantile head (independent
#                     marginals) structurally cannot get right, and the one on
#                     which the flow separates most cleanly. MEAN-INVARIANT:
#                     read only alongside CRPS/calibration, never alone.
#
# GROUP 4 -- OPERATOR-FACING FUNCTIONALS (what a day-ahead decision consumes)
#   daytotal_crps     CRPS of the day-integrated CSI: "how much total sun
#   daytotal_energy_crps  tomorrow" (and the same in integrated W/m^2 energy
#                     when clear-sky GHI is available -- the number a storage or
#                     unit-commitment optimizer actually ingests). A coherent
#                     ensemble integrates to a correct day-total distribution;
#                     independent per-step quantiles do not compose into valid
#                     day totals, so this is a place the flow should win.
#   daytotal_coverage_80 / daytotal_energy_coverage_80  calibration of that
#                     day-total interval -- an over-wide day-total band (as a
#                     quantile head tends to produce) shows up here.
#   ramp_crps         CRPS of the largest step-to-step swing in the day: "how
#                     sharp a ramp to size reserves for." A pure JOINT property
#                     (needs consecutive steps jointly right), and empirically
#                     where the flow's advantage over the quantile head is
#                     largest.
#
# GROUP 5 -- SKILL & SIGNIFICANCE (turns "A beat B" into a testable claim)
#   crps_skill_vs_ref / dm_p_vs_ref     vs the climatological CH-PeEn reference.
#   crps_skill_vs_nwp / dm_p_vs_nwp     vs NWP-direct: does conditioning on NWP
#                     beat just trusting NWP? (The check that a NWP-fed model
#                     isn't worse than its own input.)
#   *_skill_vs_deepq  vs the deep quantile baseline, on CRPS AND on every joint
#                     metric (energy, variogram, day-total, ramp). Positive here
#                     is the flow's core justification: it says the generative
#                     transport buys coherent structure the quantile head cannot,
#                     even where the two tie on marginal CRPS.
#
# skill_score = 1 - metric_model/metric_ref (positive = model better, since all
# metrics here are negatively oriented). The fair (Ferro) CRPS correction is
# used for model selection because the plain estimator's finite-ensemble bias
# grows with spread, quietly rewarding under-dispersion. diebold_mariano()
# attaches significance using a Bartlett/Newey-West variance with a lag window
# covering the window-overlap horizon.

def crps_masked(pred_ens, truth, mask, fair=False, per_step=False):

    pred_ens = np.asarray(pred_ens, np.float64)
    truth = np.asarray(truth, np.float64)
    mask = np.asarray(mask, bool)
    N, M, H = pred_ens.shape
    term1 = np.abs(pred_ens - truth[:, None, :]).mean(axis=1)     # [N,H]
    s = np.sort(pred_ens, axis=1)
    i = np.arange(M, dtype=np.float64)[None, :, None]
    pair = 2.0 * (s * (2 * i - M + 1.0)).sum(axis=1)
    denom = float(M * (M - 1)) if (fair and M > 1) else float(M * M)
    per = term1 - 0.5 * pair / denom                              # [N,H]
    per = np.where(mask, per, np.nan)
    if per_step:
        return per
    return float(np.nanmean(per))


def crps_per_window(pred_ens, truth, mask, fair=False):
    per = crps_masked(pred_ens, truth, mask, fair=fair, per_step=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(per, axis=1)


class MemberCalibrator:

    def __init__(self, K, n_days):
        self.K, self.n_days = int(K), int(n_days)
        self.a = None            # [K] additive shift per column
        self.b = None            # [K] multiplicative spread per column

    def fit(self, ens, truth, mask, b_grid=None, a_grid=None):

        ens = np.asarray(ens, np.float64)
        truth = np.asarray(truth, np.float64)
        mask = np.asarray(mask, bool)
        N, M, H = ens.shape
        K = self.K
        self.a = np.zeros(K)
        self.b = np.ones(K)
        if b_grid is None:
            b_grid = np.linspace(0.6, 1.6, 11)      # spread multipliers
        if a_grid is None:
            a_grid = np.linspace(-0.10, 0.10, 9)    # CSI bias shifts

        def col_crps(members, y, a, b):
            # members [n,M], y [n]; fair CRPS of the affine-corrected ensemble
            m = members.mean(1, keepdims=True)
            adj = a + b * (members - m) + m
            t1 = np.abs(adj - y[:, None]).mean(1)
            # fair pairwise term: sum|x_i-x_j| / (M(M-1))
            d = np.abs(adj[:, :, None] - adj[:, None, :]).sum((1, 2))
            t2 = d / (M * (M - 1))
            return float(np.mean(t1 - 0.5 * t2))

        for k in range(K):
            cols = list(range(k, H, K))
            rows = []
            ys = []
            for c in cols:
                v = mask[:, c]
                if v.any():
                    rows.append(ens[v, :, c]); ys.append(truth[v, c])
            if not rows:
                continue
            members = np.concatenate(rows, 0)       # [n,M]
            y = np.concatenate(ys, 0)               # [n]
            if members.shape[0] < 20:               # too few -> identity
                continue
            best = (0.0, 1.0, np.inf)
            # coarse joint grid, then this is already fine enough for 2 params
            for b in b_grid:
                for a in a_grid:
                    c = col_crps(members, y, a, b)
                    if c < best[2]:
                        best = (a, b, c)
            self.a[k], self.b[k] = best[0], best[1]
        return self

    def apply(self, ens):
        if self.a is None:
            return ens
        ens = np.asarray(ens, np.float64)
        N, M, H = ens.shape
        K = self.K
        out = ens.copy()
        m = ens.mean(1, keepdims=True)              # [N,1,H]
        aH = np.tile(self.a, self.n_days)[None, None, :]
        bH = np.tile(self.b, self.n_days)[None, None, :]
        out = aH + bH * (ens - m) + m
        return out

    def state(self):
        return {"K": self.K, "n_days": self.n_days,
                "a": None if self.a is None else self.a.tolist(),
                "b": None if self.b is None else self.b.tolist()}

    @classmethod
    def from_state(cls, s):
        if not s:
            return None
        c = cls(s["K"], s["n_days"])
        c.a = None if s.get("a") is None else np.asarray(s["a"], np.float64)
        c.b = None if s.get("b") is None else np.asarray(s["b"], np.float64)
        return c


def diebold_mariano(loss_a, loss_b, lag):

    d = np.asarray(loss_a, np.float64) - np.asarray(loss_b, np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dbar = float(d.mean())
    dc = d - dbar
    g0 = float((dc * dc).mean())
    var = g0
    L = int(min(max(lag, 0), n - 1))
    for k in range(1, L + 1):
        gk = float((dc[k:] * dc[:-k]).mean())
        var += 2.0 * (1.0 - k / (L + 1.0)) * gk
    var = max(var, 1e-300)
    stat = dbar / math.sqrt(var / n)
    p = float(2.0 * (1.0 - norm_cdf(np.array([abs(stat)]))[0]))
    return float(stat), p


def _masked_mean(a, mask):
    a = np.where(mask, a, np.nan)
    return float(np.nanmean(a))


def _energy_score(ens, y, valid):

    if valid.sum() < 1:
        return np.nan
    X = ens[:, valid]                                    # [M,dv]
    yv = y[valid]                                        # [dv]
    d1 = np.linalg.norm(X - yv[None, :], axis=1).mean()
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.linalg.norm(diff, axis=2).mean()
    return float(d1 - 0.5 * d2)


def _variogram_score(ens, y, valid, p=0.5):

    idx = np.where(valid)[0]
    if len(idx) < 2:
        return np.nan
    X = ens[:, idx]                                      # [M,dv]
    yv = y[idx]                                          # [dv]
    yd = np.abs(yv[:, None] - yv[None, :]) ** p          # [dv,dv]
    exd = (np.abs(X[:, :, None] - X[:, None, :]) ** p).mean(0)   # E over M
    dv = len(idx)
    return float(((yd - exd) ** 2).sum() / (dv * dv))


def _operator_metrics(pred_ens, truth, mask, K, n_days, ghi_cs=None):

    N, M, H = pred_ens.shape
    es, vs = [], []
    dt_pred, dt_true = [], []           # day-total CSI per (window,day)
    en_pred, en_true = [], []           # day-total GHI energy (if ghi_cs)
    rp_pred, rp_true = [], []           # max |ramp| per (window,day)
    have_g = ghi_cs is not None
    g = np.asarray(ghi_cs, np.float64) if have_g else None
    for i in range(N):
        for b in range(n_days):
            sl = slice(b * K, (b + 1) * K)
            v = mask[i, sl]
            if v.sum() < 1:
                continue
            ens_d = pred_ens[i, :, sl]           # [M,K]
            y_d = truth[i, sl]                    # [K]
            es.append(_energy_score(ens_d, y_d, v))
            vs.append(_variogram_score(ens_d, y_d, v))
            dt_pred.append(ens_d[:, v].sum(1))
            dt_true.append(y_d[v].sum())
            if have_g:
                gd = g[i, sl]
                gv = np.where(v & np.isfinite(gd), gd, 0.0)
                en_pred.append((ens_d * gv[None, :]).sum(1))
                en_true.append((y_d * gv).sum())
            if v.sum() >= 2:
                dif_e = np.abs(np.diff(ens_d, axis=1))          # [M,K-1]
                cons = v[:-1] & v[1:]
                if cons.any():
                    rp_pred.append(dif_e[:, cons].max(1))
                    rp_true.append(float(np.abs(np.diff(y_d))[cons].max()))

    def _mean_finite(seq):

        a = np.asarray(list(seq), np.float64)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    def _scalar_crps(pred_list, true_list):

        if not pred_list:
            return np.nan
        vals = []
        for xs, y in zip(pred_list, true_list):
            xs = np.asarray(xs)
            t1 = np.abs(xs - y).mean()
            t2 = np.abs(xs[:, None] - xs[None, :]).mean()
            vals.append(t1 - 0.5 * t2)
        return _mean_finite(vals)

    def _scalar_cov(pred_list, true_list, lo=0.1, hi=0.9):
        if not pred_list:
            return np.nan
        ins = []
        for xs, y in zip(pred_list, true_list):
            xs = np.asarray(xs)
            ins.append(float(np.quantile(xs, lo) <= y <= np.quantile(xs, hi)))
        return float(np.mean(ins))

    out = {
        # per-day scores can contain honest NaNs (1-step days have no variogram
        # pairs); _mean_finite averages the scorable days and stays silent
        "energy_score": _mean_finite(es),
        "variogram_score": _mean_finite(vs),
        "daytotal_crps": _scalar_crps(dt_pred, dt_true),
        "daytotal_coverage_80": _scalar_cov(dt_pred, dt_true),
        "ramp_crps": _scalar_crps(rp_pred, rp_true),
    }
    if have_g:
        out["daytotal_energy_crps"] = _scalar_crps(en_pred, en_true)
        out["daytotal_energy_coverage_80"] = _scalar_cov(en_pred, en_true)
    return out


def evaluate(pred_ens, truth, mask, cfg, K=None, n_days=None, ghi_cs=None):

    mask = np.asarray(mask, bool)
    truth = np.asarray(truth, np.float64)
    mean_pred = pred_ens.mean(axis=1)
    med_pred = np.median(pred_ens, axis=1)

    err2 = np.where(mask, (mean_pred - truth) ** 2, np.nan)
    rmse = float(np.sqrt(np.nanmean(err2)))
    mae = _masked_mean(np.abs(med_pred - truth), mask)
    per_step = crps_masked(pred_ens, truth, mask, per_step=True)
    crps = float(np.nanmean(per_step))

    lo = cfg["eval"].get("coverage_lo", 0.1)
    hi = cfg["eval"].get("coverage_hi", 0.9)
    ql = np.quantile(pred_ens, lo, axis=1)
    qh = np.quantile(pred_ens, hi, axis=1)
    cov = _masked_mean(((truth >= ql) & (truth <= qh)).astype(float), mask)

    # PIT-based calibration error over valid positions
    pit = (pred_ens < truth[:, None, :]).mean(axis=1)[mask]
    p = np.sort(pit)
    u = (np.arange(len(p)) + 0.5) / max(len(p), 1)
    cal = float(np.mean(np.abs(p - u))) if len(p) else float("nan")

    # enhancement tail frequency
    thr = cfg["eval"]["enhancement_threshold"]
    m3 = np.broadcast_to(mask[:, None, :], pred_ens.shape)
    ep = float((pred_ens[m3] > thr).mean())
    et = float((truth[mask] > thr).mean())

    out = {"crps": crps, "rmse": rmse, "mae": mae,
           "calibration_err": cal, "coverage_80": cov,
           "enh_frac_pred": ep, "enh_frac_true": et,
           "enh_frac_err": float(abs(ep - et)),
           "n_windows": int(truth.shape[0])}

    if ghi_cs is not None:
        g = np.asarray(ghi_cs, np.float64)
        w = np.where(mask & np.isfinite(g), g, np.nan)
        out["crps_ghi_weighted"] = float(np.nansum(w * per_step)
                                         / max(np.nansum(w), 1e-9))
        e2g = np.where(mask & np.isfinite(g),
                       ((mean_pred - truth) * g) ** 2, np.nan)
        out["rmse_ghi"] = float(np.sqrt(np.nanmean(e2g)))

    if K and n_days:
        by = {"crps": [], "rmse": [], "coverage_80": []}
        for b in range(n_days):
            sl = slice(b * K, (b + 1) * K)
            mb = mask[:, sl]
            if not mb.any():
                for v in by.values():
                    v.append(float("nan"))
                continue
            by["crps"].append(crps_masked(pred_ens[:, :, sl], truth[:, sl], mb))
            e2 = np.where(mb, (mean_pred[:, sl] - truth[:, sl]) ** 2, np.nan)
            by["rmse"].append(float(np.sqrt(np.nanmean(e2))))
            ins = (truth[:, sl] >= np.quantile(pred_ens[:, :, sl], lo, axis=1)) \
                & (truth[:, sl] <= np.quantile(pred_ens[:, :, sl], hi, axis=1))
            by["coverage_80"].append(_masked_mean(ins.astype(float), mb))
        out["crps_by_forecast_day"] = by["crps"]
        out["rmse_by_forecast_day"] = by["rmse"]
        out["coverage_by_forecast_day"] = by["coverage_80"]

        # multivariate + operator-facing joint functionals (energy score,
        # variogram score, day-total energy CRPS, ramp CRPS). These are where
        # a coherent generative ensemble is expected to beat an independent
        # per-step quantile head; the variogram score in particular targets the
        # temporal dependence structure and must be read alongside CRPS (it is
        # mean-invariant and cannot judge calibration on its own).
        out.update(_operator_metrics(pred_ens, truth, mask, K, n_days,
                                     ghi_cs=ghi_cs))
    return out


def skill_score(m_model, m_ref):
    return float("nan") if m_ref == 0 else float(1.0 - m_model / m_ref)


class DeepQuantile(FlowMatcher):

    def __init__(self, cfg, H_in, H_out, K, n_days, rep, device):
        super().__init__(cfg, H_in, H_out, K, n_days, prior=None, rep=rep,
                         device=device)
        self.q_levels: Optional[np.ndarray] = None

    # ---- training ----------------------------------------------------------
    def fit(self, hist_csi, fut_csi, hist_zen, fut_zen, hist_mask, fut_mask,
            fut_ghi_cs=None, fut_nwp=None, es_split=None, rng=None,
            n_quantiles=None, site_coords=None):
        torch, nn = _lazy_torch()
        t_cfg = self.cfg["train"]
        rng = rng or np.random.default_rng(self.cfg["seed"])
        M = int(n_quantiles or self.cfg["experiment"].get("n_ensemble", 50))
        self.q_levels = (np.arange(M, dtype=np.float32) + 0.5) / M

        # identical conditioning setup to FlowMatcher.fit
        if bool(self.cfg["model"].get("condition_clearsky_ghi", True)) \
                and fut_ghi_cs is not None:
            vals = np.asarray(fut_ghi_cs, np.float64)[np.asarray(fut_mask, bool)]
            vals = vals[np.isfinite(vals)]
            self.gcs_scale = max(1.0, float(np.percentile(vals, 99))) \
                if vals.size else None
        else:
            self.gcs_scale = None
        self.nwp_spec = self._resolve_nwp_spec(fut_nwp or {})
        self.site_vec = self._resolve_site_vec()

        self.net = _NetBuilder.build(self.cfg, self.H_in, self.H_out,
                                     self._n_fut_extra, n_out=M).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=t_cfg["lr"],
                                weight_decay=t_cfg["weight_decay"])

        y = self.rep.encode(fut_csi)                     # NaN at padding
        hc = self.rep.encode(hist_csi)
        hz = self._cos_zen(hist_zen)
        fut_ex = self._fut_extras(fut_zen, fut_mask, fut_ghi_cs, fut_nwp,
                                  site_coords=site_coords)
        hm = np.asarray(hist_mask, bool)
        fm = np.asarray(fut_mask, bool)
        hcf_np = self._fill(hc, hm)
        yf_np = self._fill(y, fm)                        # fill only feeds the
        hist_feats_np = np.stack([hcf_np, hz,            # masked loss below
                                  hm.astype(np.float32)], axis=1)

        N = y.shape[0]
        if es_split is not None and t_cfg.get("early_stopping", True):
            tr_idx, va_idx = (np.asarray(es_split[0]), np.asarray(es_split[1]))
            do_es = len(va_idx) > 0
        else:
            tr_idx, va_idx, do_es = np.arange(N), None, False

        q_t = self._to_t(self.q_levels)[None, :, None]   # [1,M,1]
        zeros_state = None                               # per-batch buffer

        def pinball(rows, train_mode=False):

            B = len(rows)
            nonlocal zeros_state
            if zeros_state is None or zeros_state.shape[0] < B:
                zeros_state = torch.zeros(B, self.H_out, device=self.device)
            x0 = zeros_state[:B]
            t05 = torch.full((B,), 0.5, device=self.device)
            hist_t = self._to_t(hist_feats_np[rows])
            hm_t = self._to_b(hm[rows])
            p_hd = float(t_cfg.get("history_dropout", 0.0) or 0.0)
            if train_mode and p_hd > 0.0:
                drop = self._to_b(rng.random(B) < p_hd)
                if drop.any():
                    keep = (~drop).float()
                    hist_t = hist_t * keep[:, None, None]
                    hm_t = hm_t & (~drop)[:, None]
            Q = self.net(x0, t05, hist_t, hm_t,
                         self._to_t(fut_ex[rows]),
                         self._to_b(fm[rows]))           # [B,M,H]
            yt = self._to_t(yf_np[rows])[:, None, :]     # [B,1,H]
            e = yt - Q
            pin = torch.maximum(q_t * e, (q_t - 1.0) * e)
            w = self._to_t(fm[rows].astype(np.float32))[:, None, :]
            return (pin * w).sum() / (w.sum() * pin.shape[1]).clamp_min(1.0)

        best, best_state, bad = np.inf, None, 0
        bs = int(t_cfg["batch_size"])
        ema = _make_ema(self.cfg, self.net)   # same EMA recipe as the flow
        for ep in range(int(t_cfg["epochs"])):
            self.net.train()
            perm = rng.permutation(tr_idx)
            running_t = torch.zeros((), device=self.device)
            nb = 0
            for s0 in range(0, len(perm), bs):
                loss = pinball(perm[s0:s0 + bs], train_mode=True)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update(self.net)
                running_t += loss.detach()
                nb += 1
            if t_cfg.get("verbose", 1) >= 1:
                print(f"  [deep_quantile] epoch {ep+1}/{t_cfg['epochs']} "
                      f"pinball={float(running_t)/max(nb,1):.5f}", flush=True)
            # ES on masked val pinball (over dense levels this is half a CRPS
            # estimate). Evaluated with the EMA weights, like the flow, so both
            # models are selected and deployed on the same footing.
            if do_es and (ep % int(t_cfg.get("es_every", 2)) == 0):
                raw_backup = ema.swap_in(self.net) if ema is not None else None
                self.net.eval()
                with torch.no_grad():
                    sel = va_idx[:int(t_cfg.get("es_max_rows", 512))]
                    vc = float(pinball(sel))
                if vc < best - float(t_cfg.get("min_delta", 1e-4)):
                    best, bad = vc, 0
                    if t_cfg.get("restore_best", True):
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                if raw_backup is not None:
                    self.net.load_state_dict(raw_backup)
                if bad >= int(t_cfg.get("patience", 15)):
                    if t_cfg.get("verbose", 1) >= 1:
                        print(f"  [deep_quantile] early stop at epoch "
                              f"{ep+1} (val pinball {best:.5f})")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        elif ema is not None:
            ema.swap_in(self.net)
        return self

    # ---- prediction ----------------------------------------------------------
    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         fut_ghi_cs=None, fut_nwp=None, n_ensemble=50, rng=None,
                         raw=False, site_coords=None):

        torch, _ = _lazy_torch()
        hc = self.rep.encode(hist_csi)
        hz = self._cos_zen(hist_zen)
        hm = np.asarray(hist_mask, bool)
        fmb = np.asarray(fut_mask, bool)
        fut_ex = self._fut_extras(np.asarray(fut_zen), fmb, fut_ghi_cs, fut_nwp,
                                  site_coords=site_coords)
        hcf = self._fill(hc, hm)
        hist_feats = np.stack([hcf, hz, hm.astype(np.float32)], axis=1)
        N = hc.shape[0]
        M = len(self.q_levels)
        chunk = max(1, int(self.cfg["train"].get("sample_chunk", 4096)) // M)
        out = np.empty((N, M, self.H_out), np.float32)
        self.net.eval()
        with torch.no_grad():
            for c0 in range(0, N, chunk):
                rows = slice(c0, min(N, c0 + chunk))
                B = rows.stop - rows.start
                x0 = torch.zeros(B, self.H_out, device=self.device)
                t05 = torch.full((B,), 0.5, device=self.device)
                Q = self.net(x0, t05, self._to_t(hist_feats[rows]),
                             self._to_b(hm[rows]), self._to_t(fut_ex[rows]),
                             self._to_b(fmb[rows]))
                Q, _ = torch.sort(Q, dim=1)              # monotone quantiles
                out[rows] = self.rep.decode(Q.cpu().numpy())
        if self.calibrator is not None and not raw:
            out = self.calibrator.apply(out)
        return out

    # ---- persistence ----------------------------------------------------------
    def save(self, path):
        torch, _ = _lazy_torch()
        ckpt = {
            "format": "deep_quantile_v1",
            "cfg": self.cfg,
            "H_in": self.H_in, "H_out": self.H_out,
            "K": self.K, "n_days": self.n_days,
            "pad_fill": self.pad_fill, "gcs_scale": self.gcs_scale,
            "nwp_spec": self.nwp_spec,
            "site_vec": (None if self.site_vec is None
                         else self.site_vec.tolist()),
            "calibrator": (None if self.calibrator is None
                           else self.calibrator.state()),
            "q_levels": self.q_levels.tolist(),
            "net_state": self.net.state_dict(),
            "rep": self.rep.state(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ckpt, path)
        return path

    @classmethod
    def load(cls, path, device=None):
        torch, _ = _lazy_torch()
        device = device or get_device("auto")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get("format") != "deep_quantile_v1":
            raise ValueError(f"{path} is not a deep_quantile checkpoint")
        cfg = ckpt["cfg"]
        rep = Representation.from_state(ckpt["rep"])
        dq = cls(cfg, ckpt["H_in"], ckpt["H_out"], ckpt["K"],
                 ckpt["n_days"], rep, device)
        dq.pad_fill = float(ckpt["pad_fill"])
        dq.gcs_scale = ckpt.get("gcs_scale")
        dq.nwp_spec = [tuple(x) for x in ckpt.get("nwp_spec", [])]
        sv = ckpt.get("site_vec")
        dq.site_vec = None if sv is None else np.asarray(sv, np.float32)
        dq.calibrator = MemberCalibrator.from_state(ckpt.get("calibrator"))
        dq.q_levels = np.asarray(ckpt["q_levels"], np.float32)
        dq.net = _NetBuilder.build(cfg, dq.H_in, dq.H_out, dq._n_fut_extra,
                                   n_out=len(dq.q_levels)).to(device)
        dq.net.load_state_dict(ckpt["net_state"])
        dq.net.eval()
        return dq


# ============================================================================
