# ===== SECTION: prior =======================================================
# ============================================================================
# Source distribution for the flow. The generic recipe is
#
#     x0 = anchor(history) + noise ,   noise ~ N(0, Sigma)
#
# with Sigma BLOCK-DIAGONAL per forecast day: correlated within each day's
# K-block, exactly zero across the night boundary (draws for different days
# are independent by construction). Within a day,
#
#     Sigma_K = D C D + jitter*I
#
# where C is a unit-diagonal stationary correlation kernel (matern12/32, rbf)
# whose length scale is solved from the masked lag-1 autocorrelation of the
# anchor residual, and D = diag(per-column residual std). The per-column
# profile makes the prior heteroscedastic: CSI variability depends strongly
# on solar-relative time of day, and the noon-centered grid is what makes a
# per-column profile meaningful in the first place.
#
# Anchors (the experiment axis):
#   white        no anchor, iid N(0, I). The uninformed control.
#   clearsky     constant clear-sky code (0 under clear-sky centering): the
#                source says "expect clear, clouds are perturbations".
#   climatology  per-column train mean profile: the source says "expect the
#                typical day at this station".
#   persistence  repeat the last observed history day across the forecast
#                days: the source carries today's weather into tomorrow.
#   blend        per forecast day d, w_d * persistence + (1-w_d) * climatology,
#                with w_d = the train correlation between the last history
#                day's mean and forecast-day-d's mean. Day-to-day cloud memory
#                decays fast; this anchor decays with it instead of tiling the
#                last day unchanged into day 2 and beyond.
#   nwp          (B) anchor on the NWP's OWN clear-sky-index forecast for the
#                horizon (DSWRF / ghi_clearsky, encoded to model space). Unlike
#                every anchor above -- which are built from the station's own
#                history or its climatology -- this one is exogenous: it injects
#                the numerical weather model's day-ahead cloud guess as the
#                source mean, so the flow only has to transport away the NWP's
#                ERROR rather than the full distance from clear sky. The
#                anchor is supplied per window to fit_for()/sample() as
#                `fut_nwp_anchor` (model space, padding-filled); the fitted
#                per-column std and length scale then describe NWP forecast
#                error, not clear-sky spread. If no NWP-CSI channel is present
#                for a batch, it degrades gracefully to the clearsky anchor.
#                Requires data.nwp.enabled + derive_csi and a window file
#                carrying fut_nwp_csi (built in preprocess). This is the prior
#                counterpart to (A), which feeds the same NWP fields as extra
#                horizon CONDITIONING channels; the two can be used together or
#                compared as separate experiment cells.
#
# Anchors above the nwp line are endogenous (history/climatology); nwp is the
# one exogenous anchor. Both (A) and (B) keep NWP strictly on the horizon side.
#
# Optional extras (config prior.*, apply to any correlated anchor):
#   enhancement  with probability w, add a one-sided short-length-scale burst,
#                putting source mass above clear sky where the data has its
#                enhancement tail. w = 'auto' measures it from train.
#   regime       scale the noise by the residual variability of the sky regime
#                (clear/broken/overcast) of the most recent history day.
#
# Mask handling: padded positions receive no noise and sit exactly at the
# anchor, so every draw is finite; the loss and metrics ignore those slots
# anyway. Sampling never needs a Cholesky of the full H x H matrix: one K x K
# factor is reused per day block, which is also what guarantees exact zeros
# across nights.

PRIOR_KINDS = ("white", "clearsky", "climatology", "persistence", "blend",
               "nwp")
_UNSET = object()


def _corr(kind, dlag, l):
    l = max(float(l), 1e-3)
    if kind == "rbf":
        return np.exp(-0.5 * (dlag / l) ** 2)
    if kind == "matern12":
        return np.exp(-dlag / l)
    if kind == "matern32":
        r = np.sqrt(3.0) * dlag / l
        return (1.0 + r) * np.exp(-r)
    raise ValueError(f"unknown kernel '{kind}'")


def _corr_matrix(kind, K, length_scale):
    idx = np.arange(K)
    dlag = np.abs(idx[:, None] - idx[None, :]).astype(float)
    return _corr(kind, dlag, length_scale)


def _solve_length_scale(kind, rho1, K):
    lo, hi = 0.05, float(max(K, 2))
    f = lambda l: float(_corr(kind, np.array([1.0]), l)[0])
    if rho1 <= f(lo):
        return lo
    if rho1 >= f(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < rho1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _masked_lag1(resid, fut_mask, K):

    resid = np.asarray(resid, np.float64)
    fm = np.asarray(fut_mask, bool)
    H = resid.shape[1]
    n_days = H // K
    num = den = 0.0
    for b in range(n_days):
        blk = resid[:, b*K:(b+1)*K]
        bm = fm[:, b*K:(b+1)*K]
        if not bm.any():
            continue
        mu = float(np.nanmean(np.where(bm, blk, np.nan)))
        mu = 0.0 if not np.isfinite(mu) else mu
        blk = blk - mu
        a, c = blk[:, :-1], blk[:, 1:]
        pair = bm[:, :-1] & bm[:, 1:] & np.isfinite(a) & np.isfinite(c)
        num += float((a[pair] * c[pair]).sum())
        den += float((blk[bm & np.isfinite(blk)] ** 2).sum())
    if den <= 0:
        return 0.9
    return float(np.clip(num / den, 1e-3, 0.999))


class DayPrior:

    def __init__(self, kind, K, n_days, cfg):
        assert kind in PRIOR_KINDS, f"unknown prior '{kind}'"
        self.kind = kind
        self.K = int(K)
        self.n_days = int(n_days)
        self.H = self.K * self.n_days
        self.cfg = cfg
        # fitted state
        self._L = None                      # [K,K] Cholesky of Sigma_K
        self._Le = None                     # [K,K] Cholesky of the burst kernel
        self._clearsky_code = _UNSET
        self._clim_cols = None              # [K] per-column train mean (model space)
        self._blend_w = None                # [n_days] persistence weights
        self._enh_weight = 0.0
        self._regime_mult = None            # {'clear':a,'broken':b,'overcast':c}
        self._fit_info: Dict[str, Any] = {}

    # ---- small properties ----------------------------------------------------
    @property
    def clearsky_code(self):
        if self._clearsky_code is _UNSET:
            raise RuntimeError("clearsky_code unset; call fit_for() first")
        return float(self._clearsky_code)

    @clearsky_code.setter
    def clearsky_code(self, v):
        self._clearsky_code = float(v)

    @property
    def needs_history(self):
        return self.kind in ("persistence", "blend") or \
            bool((self.cfg["prior"].get("regime") or {}).get("enabled"))

    @property
    def needs_nwp(self):
        # (B) the nwp-anchored prior needs a per-window NWP-CSI anchor (in model
        # space) supplied to fit_for/sample; without it, it degrades to clearsky
        return self.kind == "nwp"

    @property
    def is_correlated(self):
        return self.kind != "white"

    def full_covariance(self):

        if self._L is None:
            return np.eye(self.H)
        S = np.zeros((self.H, self.H))
        Sk = self._L @ self._L.T
        for b in range(self.n_days):
            S[b*self.K:(b+1)*self.K, b*self.K:(b+1)*self.K] = Sk
        return S

    # ---- fitting ---------------------------------------------------------------
    def fit_for(self, x1_train, hist_rep, hist_mask, fut_mask, clearsky_code,
                hist_csi_phys=None, enh_frac_train=None, fut_nwp_anchor=None):


        self.clearsky_code = clearsky_code
        # keep the training NWP anchor only to validate shapes; the anchor used
        # at sample time is always the per-batch one passed to sample()
        self._nwp_anchor_seen = fut_nwp_anchor is not None
        x1 = np.asarray(x1_train, np.float64)
        fm = np.asarray(fut_mask, bool)
        hm = np.asarray(hist_mask, bool)
        hr = np.asarray(hist_rep, np.float64)

        # per-column climatology in model space (fallback: clear-sky code)
        clim = np.full(self.K, self.clearsky_code)
        for k in range(self.K):
            vals = x1[:, k::self.K][fm[:, k::self.K]]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                clim[k] = float(vals.mean())
        self._clim_cols = clim

        # blend weights: correlation between the last history day's mean and
        # each forecast day's mean, over windows where both are observed
        w = np.zeros(self.n_days)
        h_last = np.where(hm[:, -self.K:], hr[:, -self.K:], np.nan)
        h_mean = np.nanmean(h_last, axis=1)
        for dday in range(self.n_days):
            blk = np.where(fm[:, dday*self.K:(dday+1)*self.K],
                           x1[:, dday*self.K:(dday+1)*self.K], np.nan)
            f_mean = np.nanmean(blk, axis=1)
            ok = np.isfinite(h_mean) & np.isfinite(f_mean)
            if ok.sum() >= 10 and np.nanstd(h_mean[ok]) > 0 \
                    and np.nanstd(f_mean[ok]) > 0:
                r = float(np.corrcoef(h_mean[ok], f_mean[ok])[0, 1])
                w[dday] = float(np.clip(r, 0.0, 1.0))
            else:
                w[dday] = 0.5 ** (dday + 1)
        self._blend_w = w

        if not self.is_correlated:
            self._fit_info = {"kind": "white"}
            return self

        # anchor residual, valid positions only. The 'nwp' prior anchors on the
        # per-window NWP-CSI forecast (model space) rather than a history-derived
        # anchor, so its residual statistics capture NWP forecast error.
        if self.kind == "nwp" and fut_nwp_anchor is not None:
            anchor = np.asarray(fut_nwp_anchor, np.float64)
        else:
            anchor = self._anchor(hr, hm)
        resid = x1 - anchor
        valid = fm & np.isfinite(resid)
        pooled_std = float(np.std(resid[valid])) if valid.any() else 1.0
        pooled_std = max(pooled_std, 1e-4)

        # heteroscedastic per-column std, floored for never-seen edge columns
        pc = self.cfg["prior"]
        floor = float(pc.get("col_std_floor_frac", 0.10)) * pooled_std
        col_std = np.full(self.K, pooled_std)
        for k in range(self.K):
            vals = resid[:, k::self.K][valid[:, k::self.K]]
            if vals.size >= 10:
                col_std[k] = float(np.std(vals))
        col_std = np.maximum(col_std, floor)

        # correlation length from the masked lag-1 autocorrelation of the
        # STANDARDIZED residual (so the heteroscedasticity does not bleed
        # into the correlation estimate)
        std_tiled = np.tile(col_std, self.n_days)
        rho1 = _masked_lag1(resid / std_tiled[None, :], fm, self.K)
        ls = _solve_length_scale(pc["kernel"], rho1, self.K)

        C = _corr_matrix(pc["kernel"], self.K, ls)
        Sk = (col_std[:, None] * C * col_std[None, :]) \
            + float(pc["jitter"]) * np.eye(self.K)
        self._L = np.linalg.cholesky(Sk)

        # one-sided enhancement burst
        ec = pc.get("enhancement") or {}
        wcfg = ec.get("weight", 0.0)
        if wcfg == "auto":
            self._enh_weight = float(enh_frac_train or 0.0)
        else:
            self._enh_weight = float(wcfg or 0.0)
        if self._enh_weight > 0:
            amp = float(ec.get("amplitude_frac", 0.5)) * pooled_std
            Ce = _corr_matrix(pc["kernel"], self.K,
                              float(ec.get("length_scale", 1.0)))
            Se = (amp ** 2) * Ce + float(pc["jitter"]) * np.eye(self.K)
            self._Le = np.linalg.cholesky(Se)

        # regime multipliers from residual variability conditioned on the
        # last history day's sky state
        rc = pc.get("regime") or {}
        if rc.get("enabled") and hist_csi_phys is not None:
            lab = self._classify(np.asarray(hist_csi_phys), hm)
            mults = {}
            lo_clip, hi_clip = rc.get("mult_clip", [0.5, 2.0])
            for name in ("clear", "broken", "overcast"):
                rows = lab == name
                v = resid[rows][valid[rows]] if rows.any() else np.array([])
                if v.size >= 50:
                    m = float(np.std(v) / pooled_std)
                else:
                    m = 1.0
                mults[name] = float(np.clip(m, lo_clip, hi_clip))
            self._regime_mult = mults

        self._fit_info = {"kind": self.kind, "rho1": rho1, "length_scale": ls,
                          "pooled_std": pooled_std,
                          "col_std_min": float(col_std.min()),
                          "col_std_max": float(col_std.max()),
                          "enh_weight": self._enh_weight,
                          "blend_w": w.tolist(),
                          "regime_mult": self._regime_mult}
        return self

    def _classify(self, hist_csi_phys, hist_mask):
        rc = self.cfg["prior"].get("regime") or {}
        last = np.where(hist_mask[:, -self.K:], hist_csi_phys[:, -self.K:], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(last, axis=1)
            s = np.nanstd(last, axis=1)
        lab = np.full(len(m), "broken", dtype=object)
        clear = (m >= float(rc.get("clear_mean_min", 0.90))) & \
                (s <= float(rc.get("clear_std_max", 0.08)))
        overcast = m <= float(rc.get("overcast_mean_max", 0.45))
        lab[clear] = "clear"
        lab[overcast & ~clear] = "overcast"
        lab[~np.isfinite(m)] = "broken"
        return lab

    # ---- anchors ---------------------------------------------------------------
    def _anchor(self, hist_rep, hist_mask, fut_nwp_anchor=None):
        N = hist_rep.shape[0]
        clim_tiled = np.tile(self._clim_cols, self.n_days)
        if self.kind == "nwp":
            # (B) anchor on the supplied per-window NWP-CSI forecast; if it is
            # missing (NWP unavailable for these rows), fall back to clear sky
            # so the prior stays well-defined rather than crashing.
            if fut_nwp_anchor is not None:
                a = np.asarray(fut_nwp_anchor, np.float64)
                assert a.shape == (N, self.H), \
                    f"nwp anchor shape {a.shape} != {(N, self.H)}"
                return np.where(np.isfinite(a), a, self.clearsky_code)
            return np.full((N, self.H), self.clearsky_code)
        if self.kind in ("white", "clearsky"):
            return np.full((N, self.H), self.clearsky_code)
        if self.kind == "climatology":
            return np.broadcast_to(clim_tiled, (N, self.H)).copy()
        # persistence / blend both start from the last observed day; padded
        # history steps fall back to the climatology column, which is a far
        # better stand-in than pretending the padded step was clear
        last = hist_rep[:, -self.K:].astype(np.float64)
        lm = hist_mask[:, -self.K:].astype(bool)
        last = np.where(lm, last, self._clim_cols[None, :])
        persist = np.tile(last, (1, self.n_days))
        if self.kind == "persistence":
            return persist
        # blend
        out = np.empty((N, self.H))
        for dday in range(self.n_days):
            wd = float(self._blend_w[dday])
            sl = slice(dday*self.K, (dday+1)*self.K)
            out[:, sl] = wd * persist[:, sl] + (1 - wd) * self._clim_cols[None, :]
        return out

    # ---- sampling ---------------------------------------------------------------
    def sample(self, n, rng, hist_rep=None, hist_mask=None, fut_mask=None,
               hist_csi_phys=None, fut_nwp_anchor=None):

        if fut_mask is not None:
            fmx = np.asarray(fut_mask)
            assert fmx.shape == (n, self.H), \
                f"fut_mask shape {fmx.shape} != {(n, self.H)}"
        if self.kind != "white" or self.needs_history:
            assert hist_rep is not None and hist_mask is not None, \
                f"prior '{self.kind}' needs history"
            assert np.asarray(hist_rep).shape[0] == n

        if self.kind == "white":
            noise = rng.standard_normal((n, self.H)).astype(np.float32)
            anchor = np.zeros((n, self.H), np.float32)
        else:
            eps = rng.standard_normal((n, self.n_days, self.K))
            noise = np.einsum("nbk,jk->nbj", eps, self._L).reshape(n, self.H)
            if self._enh_weight > 0 and self._Le is not None:
                fire = rng.random(n) < self._enh_weight
                if fire.any():
                    ee = rng.standard_normal((int(fire.sum()), self.n_days, self.K))
                    burst = np.abs(np.einsum("nbk,jk->nbj", ee, self._Le))
                    noise[fire] += burst.reshape(-1, self.H)
            if self._regime_mult is not None and hist_csi_phys is not None:
                lab = self._classify(np.asarray(hist_csi_phys),
                                     np.asarray(hist_mask, bool))
                mult = np.array([self._regime_mult.get(x, 1.0) for x in lab])
                noise *= mult[:, None]
            noise = noise.astype(np.float32)
            anchor = self._anchor(np.asarray(hist_rep, np.float64),
                                  np.asarray(hist_mask, bool),
                                  fut_nwp_anchor=fut_nwp_anchor).astype(np.float32)

        out = anchor + noise
        # padded slots carry no noise and sit exactly at a finite value
        if fut_mask is not None:
            fmx = np.asarray(fut_mask, bool)
            safe = np.where(np.isfinite(anchor), anchor, self.clearsky_code)
            out = np.where(fmx, out, safe).astype(np.float32)
        return out

    # ---- (de)serialization --------------------------------------------------
    def state(self):
        return {"kind": self.kind, "K": self.K, "n_days": self.n_days,
                "clearsky_code": (None if self._clearsky_code is _UNSET
                                  else float(self._clearsky_code)),
                "L": self._L, "Le": self._Le,
                "clim_cols": self._clim_cols, "blend_w": self._blend_w,
                "enh_weight": self._enh_weight,
                "regime_mult": self._regime_mult, "fit_info": self._fit_info}

    @classmethod
    def from_state(cls, s, cfg):
        p = cls(s["kind"], s["K"], s["n_days"], cfg)
        if s["clearsky_code"] is not None:
            p.clearsky_code = s["clearsky_code"]
        p._L = None if s["L"] is None else np.asarray(s["L"])
        p._Le = None if s["Le"] is None else np.asarray(s["Le"])
        p._clim_cols = None if s["clim_cols"] is None else np.asarray(s["clim_cols"])
        p._blend_w = None if s["blend_w"] is None else np.asarray(s["blend_w"])
        p._enh_weight = float(s.get("enh_weight", 0.0))
        p._regime_mult = s.get("regime_mult")
        p._fit_info = s.get("fit_info", {})
        return p

    def describe(self):
        return dict(self._fit_info)


def make_prior(cfg, kind, K, n_days):
    return DayPrior(kind, K, n_days, cfg)


# ============================================================================
