# ===== SECTION: baselines ===================================================
# ============================================================================
# The reference LADDER every flow cell must climb. All share the same
# predict_ensemble signature as the flow so one evaluate() covers everything,
# and all are mask-aware. Where a baseline must place a value at a horizon step
# whose matching history step is padded (a longer future day), it falls back to
# the per-column train median rather than inventing a clear sky.
#
# The ladder, from weakest to strongest, and WHY each rung exists:
#
#   day_persistence : tomorrow = today. Deterministic; degenerate ensemble.
#                     The classic hard-to-beat point reference. If a model
#                     loses to this, something is broken.
#   peen            : persistence ensemble; the last `peen_days` observed days
#                     are the members. The standard cheap probabilistic
#                     reference -- adds spread to persistence.
#   ch_peen         : complete-history per-column empirical distribution of
#                     CSI from the training period (CH-PeEn, Yang 2019). THE
#                     climatology reference skill scores are quoted against in
#                     the solar literature; eval.dm_reference points here.
#   analog_day      : probabilistic persistence; samples whole future days
#                     that historically followed a similar last day. The
#                     strongest HISTORY-ONLY reference: coherent real days,
#                     conditioned on the current regime.
#   blend           : BLEND persistence benchmark. It combines scalar
#                     persistence with yesterday's day profile using a
#                     training-only least-squares weight clipped to [0, 1].
#   blend_corr      : correlation-weighted BLEND, retained as a separate
#                     reproducible comparator rather than a tuned model.
#   nwp_direct      : (in SECTION run, class NWPDirect near evaluate) the raw
#                     HRRR NWP-CSI forecast itself, scored on the identical
#                     masked cells. EXOGENOUS reference: the bar any model
#                     that CONSUMES NWP must clear -- a learned model can
#                     silently be worse than its own input, and this row is
#                     how we would know. Registered automatically whenever
#                     the windows carry fut_nwp_csi.
#   deep_quantile   : (class DeepQuantile, defined after FlowMatcher) same
#                     backbone + conditioning as the flow, multi-quantile
#                     pinball head. The DEEP reference that isolates the
#                     generative transport: flow-vs-deepq differences are
#                     attributable to the transport alone. Expect it to edge
#                     the flow on marginal CRPS and lose on the joint metrics
#                     -- that asymmetry IS the finding.

def _col_median(fut_csi, fut_mask, K):
    fut_csi = np.asarray(fut_csi, np.float64)
    fm = np.asarray(fut_mask, bool)
    med = np.full(K, 1.0)
    allv = fut_csi[fm]
    global_med = float(np.median(allv)) if allv.size else 1.0
    for k in range(K):
        v = fut_csi[:, k::K][fm[:, k::K]]
        med[k] = float(np.median(v)) if v.size else global_med
    return med


class DayPersistence:
    def __init__(self, K, n_days):
        self.K, self.n_days = int(K), int(n_days)
        self._med = None

    def fit(self, fut_csi, fut_mask):
        self._med = _col_median(fut_csi, fut_mask, self.K)
        return self

    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         fut_ghi_cs=None, fut_nwp=None, n_ensemble=1, rng=None):
        last = np.asarray(hist_csi)[:, -self.K:].astype(np.float64)
        lm = np.asarray(hist_mask, bool)[:, -self.K:]
        last = np.where(lm, np.nan_to_num(last, nan=1.0), self._med[None, :])
        pred = np.tile(last, (1, self.n_days)).astype(np.float32)
        return np.repeat(pred[:, None, :], n_ensemble, axis=1)


class BlendPersistence:
    """Blend ordinary persistence with cyclic day-profile persistence.

    The fit uses only the training windows of the current fold. The least-squares
    variant follows Cyril Voyant's benchmark: the blend weight minimizes training
    squared error and is clipped to [0, 1]. The correlation variant uses the
    correlation-based weight as a separate reproducible reference.
    """

    def __init__(self, K, n_days, mode="least_squares"):
        self.K = int(K)
        self.n_days = int(n_days)
        self.mode = str(mode)
        self.weight = 0.5
        self._med = None

    def fit(self, hist_csi, fut_csi, hist_mask, fut_mask):
        hist = np.asarray(hist_csi, np.float64)
        hm = np.asarray(hist_mask, bool)
        fut = np.asarray(fut_csi, np.float64)
        fm = np.asarray(fut_mask, bool)

        self._med = _col_median(fut, fm, self.K)
        last = hist[:, -self.K:]
        last_mask = hm[:, -self.K:]
        cyclic = np.where(
            last_mask,
            np.nan_to_num(last, nan=1.0),
            self._med[None, :],
        )

        # Ordinary persistence carries the last observed CSI value through
        # the forecast. Cyclic persistence carries yesterday's profile.
        last_value = np.where(
            hm[:, -1] & np.isfinite(hist[:, -1]),
            hist[:, -1],
            np.nan,
        )
        fallback = np.nanmedian(self._med)
        last_value = np.where(np.isfinite(last_value), last_value, fallback)
        ordinary = np.repeat(last_value[:, None], self.K, axis=1)

        target = fut[:, :self.K]
        valid = fm[:, :self.K] & np.isfinite(target)
        valid &= np.isfinite(cyclic) & np.isfinite(ordinary)
        if not valid.any():
            self.weight = 0.5
            return self

        p = ordinary[valid]
        c = cyclic[valid]
        y = target[valid]

        if self.mode == "correlation":
            if p.size < 2 or np.std(p) < 1e-12 or np.std(c) < 1e-12:
                self.weight = 0.5
            else:
                rho = float(np.corrcoef(p, c)[0, 1])
                self.weight = float(np.clip(0.5 * (1.0 + rho), 0.0, 1.0))
        else:
            d = p - c
            den = float(np.dot(d, d))
            self.weight = (
                float(np.clip(np.dot(d, y - c) / den, 0.0, 1.0))
                if den > 1e-12 else 0.5
            )
        return self

    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask,
                         fut_mask, fut_ghi_cs=None, fut_nwp=None,
                         n_ensemble=1, rng=None):
        hist = np.asarray(hist_csi, np.float64)
        hm = np.asarray(hist_mask, bool)
        last = hist[:, -self.K:]
        lm = hm[:, -self.K:]
        cyclic = np.where(lm, np.nan_to_num(last, nan=1.0), self._med[None, :])
        last_value = np.where(
            hm[:, -1] & np.isfinite(hist[:, -1]),
            hist[:, -1],
            np.nan,
        )
        fallback = np.nanmedian(self._med)
        last_value = np.where(np.isfinite(last_value), last_value, fallback)
        ordinary = np.repeat(last_value[:, None], self.K, axis=1)
        day = self.weight * ordinary + (1.0 - self.weight) * cyclic
        pred = np.tile(day, (1, self.n_days)).astype(np.float32)
        return np.repeat(pred[:, None, :], int(n_ensemble), axis=1)


class PeEn:
    def __init__(self, K, n_days, m_days):
        self.K, self.n_days = int(K), int(n_days)
        self.m_days = int(m_days)
        self._med = None

    def fit(self, fut_csi, fut_mask):
        self._med = _col_median(fut_csi, fut_mask, self.K)
        return self

    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         fut_ghi_cs=None, fut_nwp=None, n_ensemble=50, rng=None):
        hist_csi = np.asarray(hist_csi, np.float64)
        hm = np.asarray(hist_mask, bool)
        Hd = hist_csi.shape[1] // self.K
        m = min(self.m_days, Hd)
        members = []
        for j in range(m):                      # j = 0 is the most recent day
            sl = slice((Hd - 1 - j) * self.K, (Hd - j) * self.K)
            day = np.where(hm[:, sl], np.nan_to_num(hist_csi[:, sl], nan=1.0),
                           self._med[None, :])
            members.append(np.tile(day, (1, self.n_days)))
        mem = np.stack(members, axis=1).astype(np.float32)   # [N,m,H]
        reps = int(np.ceil(n_ensemble / m))
        return np.tile(mem, (1, reps, 1))[:, :n_ensemble, :]


class CHPeEn:

    def __init__(self, K, n_days):
        self.K, self.n_days = int(K), int(n_days)
        self.pool: Dict[int, np.ndarray] = {}
        self._global = None

    def fit(self, fut_csi, fut_mask):
        fut_csi = np.asarray(fut_csi, np.float64)
        fm = np.asarray(fut_mask, bool)
        for k in range(self.K):
            v = fut_csi[:, k::self.K][fm[:, k::self.K]]
            if v.size:
                self.pool[k] = v.astype(np.float32)
        self._global = fut_csi[fm].astype(np.float32)
        if self._global.size == 0:
            raise ValueError("CHPeEn.fit saw no valid values")
        return self

    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         fut_ghi_cs=None, fut_nwp=None, n_ensemble=50, rng=None):
        rng = rng or np.random.default_rng(0)
        N, H = np.asarray(fut_mask).shape
        out = np.empty((N, n_ensemble, H), np.float32)
        for h in range(H):
            pool = self.pool.get(h % self.K, self._global)
            out[:, :, h] = rng.choice(pool, size=(N, n_ensemble))
        return out


class NWPDirect:

    def __init__(self, K, n_days, spread_csi=0.0):
        self.K, self.n_days = int(K), int(n_days)
        self.spread_csi = float(spread_csi)
        self._med = None

    def fit(self, fut_csi, fut_mask):
        # per-column climatology only used to fill cells with no NWP value
        self._med = _col_median(fut_csi, fut_mask, self.K)
        return self

    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         fut_ghi_cs=None, fut_nwp=None, n_ensemble=50, rng=None):
        rng = rng or np.random.default_rng(0)
        N, H = np.asarray(fut_mask).shape
        # locate the NWP-CSI channel; if absent, this baseline can't run
        arr = None if not fut_nwp else fut_nwp.get("fut_nwp_csi")
        if arr is None:
            raise ValueError(
                "NWPDirect needs the fut_nwp_csi channel (data.nwp.derive_csi "
                "must be on and the windows must carry it). Disable the "
                "nwp_direct baseline or rebuild windows with NWP-CSI.")
        base = np.asarray(arr, np.float64).copy()          # [N,H]
        # fill missing NWP cells with the per-column climatology
        clim = np.tile(self._med, self.n_days)[None, :]
        base = np.where(np.isfinite(base), base, clim)
        pred = np.repeat(base[:, None, :], n_ensemble, axis=1).astype(np.float32)
        if self.spread_csi > 0 and n_ensemble > 1:
            pred = pred + rng.normal(0.0, self.spread_csi,
                                     size=pred.shape).astype(np.float32)
        return pred


class AnalogDay:


    def __init__(self, K, n_days, n_bins=20):
        self.K, self.n_days, self.n_bins = int(K), int(n_days), int(n_bins)
        self.edges = None
        self.pools: Dict[int, np.ndarray] = {}
        self._global = None
        self._med = None

    def _last_mean(self, hist_csi, hist_mask):
        m = np.where(np.asarray(hist_mask, bool)[:, -self.K:],
                     np.asarray(hist_csi, np.float64)[:, -self.K:], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(m, axis=1)

    def fit(self, hist_csi, fut_csi, hist_mask, fut_mask):
        self._med = _col_median(fut_csi, fut_mask, self.K)
        # analog members must be finite everywhere they might be scored, so
        # padded slots take the column median
        fut = np.where(np.asarray(fut_mask, bool),
                       np.nan_to_num(np.asarray(fut_csi, np.float64), nan=1.0),
                       np.tile(self._med, np.asarray(fut_csi).shape[1] // self.K)[None, :])
        self._global = fut.astype(np.float32)
        lm = self._last_mean(hist_csi, hist_mask)
        ok = np.isfinite(lm)
        lo, hi = np.percentile(lm[ok], [1, 99]) if ok.any() else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1e-6
        self.edges = np.linspace(lo, hi, self.n_bins + 1)
        b = np.clip(np.digitize(lm, self.edges) - 1, 0, self.n_bins - 1)
        for bi in range(self.n_bins):
            rows = np.where(ok & (b == bi))[0]
            if rows.size:
                self.pools[bi] = self._global[rows]
        return self

    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         fut_ghi_cs=None, fut_nwp=None, n_ensemble=50, rng=None):
        rng = rng or np.random.default_rng(0)
        lm = self._last_mean(hist_csi, hist_mask)
        lm = np.where(np.isfinite(lm), lm, float(np.nanmedian(self.edges)))
        b = np.clip(np.digitize(lm, self.edges) - 1, 0, self.n_bins - 1)
        N, H = np.asarray(fut_mask).shape
        out = np.empty((N, n_ensemble, H), np.float32)
        for i in range(N):
            pool = self.pools.get(int(b[i]), self._global)
            if pool.shape[0] == 0:
                pool = self._global
            pick = rng.integers(0, pool.shape[0], size=n_ensemble)
            out[i] = pool[pick]
        return out


# ============================================================================
