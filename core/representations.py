# ===== SECTION: represent ===================================================
# ============================================================================
# Maps physical CSI <-> model space. The full map is
#
#     z = (rep(csi) - center) / scale
#
# with `rep` one of four monotone transforms, and center/scale fit on TRAIN
# valid values only. Under center='clearsky' the code of CSI = 1 is exactly 0,
# which is what makes "clouds are perturbations from zero" literally true for
# the clear-sky-anchored prior. The self-test asserts the round trip.
#
# Why four transforms are on the experiment axis:
#
#   raw    identity. The control.
#   log    rep = log(csi). The Beer-Lambert coordinate: for beam irradiance,
#          optical depth is additive in this space. GHI mixes in diffuse
#          radiation and enhancement (csi > 1), which breaks the additivity,
#          so treat this as a physically motivated hypothesis to test, not a
#          known-good default.
#   logit  a scaled logit on (csi_min, csi_max). Respects the hard physical
#          bounds, so the flow cannot place probability mass outside them and
#          decode-time clipping never distorts the ensemble CDF at the edges.
#   gauss  empirical Gaussianization: csi -> empirical train CDF -> normal
#          score. Marginals become N(0,1) by construction, absorbing the
#          bimodality (clear mode near 1, cloudy mode below) and the skew that
#          defeat any fixed analytic transform. The flow then only has to
#          learn temporal dependence, and the Gaussian prior matches the true
#          marginals exactly. Monotone and invertible by interpolation over
#          fitted quantile knots.
#
# Padding is orthogonal to all of this: transforms act on VALUES, the mask
# says which POSITIONS are real, and encode/decode propagate NaN unchanged.

REPRESENTATION_KINDS = ("raw", "log", "logit", "gauss")


class Representation:
    def __init__(self, kind: str, center: str, csi_min: float, csi_max: float,
                 rep_cfg: Optional[Dict[str, Any]] = None):
        assert kind in REPRESENTATION_KINDS, f"unknown representation '{kind}'"
        assert center in ("clearsky", "mean")
        assert csi_min > 0.0, "csi_min must be > 0 (log transform)"
        self.kind = kind
        self.center_mode = center
        self.csi_min = float(csi_min)
        self.csi_max = float(csi_max)
        rc = rep_cfg or {}
        self.gauss_knots = int(rc.get("gauss_knots", 1001))
        self.gauss_p_clip = float(rc.get("gauss_p_clip", 1e-6))
        self.logit_margin = float(rc.get("logit_margin", 1e-4))
        # fitted state
        self.center = 0.0
        self.scale = 1.0
        self._knot_v: Optional[np.ndarray] = None   # gauss: csi knots
        self._knot_p: Optional[np.ndarray] = None   # gauss: CDF levels

    # ---- forward / inverse of rep(), NaN-transparent -----------------------
    def _to_rep(self, csi: np.ndarray) -> np.ndarray:
        csi = np.asarray(csi, dtype=np.float64)
        out = np.full(csi.shape, np.nan)
        ok = np.isfinite(csi)
        if not ok.any():
            return out
        c = np.clip(csi[ok], self.csi_min, self.csi_max)
        if self.kind == "raw":
            out[ok] = c
        elif self.kind == "log":
            out[ok] = np.log(c)
        elif self.kind == "logit":
            u = (c - self.csi_min) / (self.csi_max - self.csi_min)
            u = np.clip(u, self.logit_margin, 1.0 - self.logit_margin)
            out[ok] = np.log(u / (1.0 - u))
        elif self.kind == "gauss":
            assert self._knot_v is not None, "gauss transform used before fit()"
            p = np.interp(c, self._knot_v, self._knot_p)
            p = np.clip(p, self.gauss_p_clip, 1.0 - self.gauss_p_clip)
            out[ok] = norm_ppf(p)
        return out

    def _from_rep(self, rep: np.ndarray) -> np.ndarray:
        rep = np.asarray(rep, dtype=np.float64)
        out = np.full(rep.shape, np.nan)
        ok = np.isfinite(rep)
        if not ok.any():
            return out
        r = rep[ok]
        if self.kind == "raw":
            c = r
        elif self.kind == "log":
            c = np.exp(r)
        elif self.kind == "logit":
            u = 1.0 / (1.0 + np.exp(-r))
            c = self.csi_min + u * (self.csi_max - self.csi_min)
        elif self.kind == "gauss":
            p = norm_cdf(r)
            c = np.interp(p, self._knot_p, self._knot_v)
        out[ok] = np.clip(c, self.csi_min, self.csi_max)
        return out

    def _clearsky_rep(self) -> float:
        return float(self._to_rep(np.array([1.0]))[0])

    # ---- fitting ------------------------------------------------------------
    def fit(self, csi_train: np.ndarray) -> "Representation":

        v = np.asarray(csi_train, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < 10:
            raise ValueError("Representation.fit needs at least 10 valid values")

        if self.kind == "gauss":
            # empirical quantile knots at interior plotting positions
            # (i+0.5)/m, so the physical-bound anchors added below extend the
            # CDF strictly (p=0 and p=1 stay unique). Ties in the data are
            # deduplicated so np.interp sees a strictly increasing abscissa
            # in both directions.
            m = self.gauss_knots
            levels = (np.arange(m) + 0.5) / m
            knots = np.quantile(np.clip(v, self.csi_min, self.csi_max), levels)
            kv, idx = np.unique(knots, return_index=True)
            kp = levels[idx]
            # anchor the ends at the physical bounds so every in-range value
            # interpolates rather than saturating
            if kv[0] > self.csi_min:
                kv = np.concatenate([[self.csi_min], kv])
                kp = np.concatenate([[0.0], kp])
            if kv[-1] < self.csi_max:
                kv = np.concatenate([kv, [self.csi_max]])
                kp = np.concatenate([kp, [1.0]])
            self._knot_v, self._knot_p = kv, kp

        rep = self._to_rep(v)
        rep = rep[np.isfinite(rep)]
        self.scale = float(rep.std() + 1e-8)
        self.center = self._clearsky_rep() if self.center_mode == "clearsky" \
            else float(rep.mean())
        return self

    # ---- public encode/decode ------------------------------------------------
    def encode(self, csi: np.ndarray) -> np.ndarray:
        return ((self._to_rep(csi) - self.center) / self.scale).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        rep = np.asarray(z, dtype=np.float64) * self.scale + self.center
        return self._from_rep(rep).astype(np.float32)

    def clearsky_code(self) -> float:
        return float((self._clearsky_rep() - self.center) / self.scale)

    def state(self) -> Dict[str, Any]:
        return {"kind": self.kind, "center_mode": self.center_mode,
                "csi_min": self.csi_min, "csi_max": self.csi_max,
                "center": self.center, "scale": self.scale,
                "gauss_knots": self.gauss_knots,
                "gauss_p_clip": self.gauss_p_clip,
                "logit_margin": self.logit_margin,
                "knot_v": None if self._knot_v is None else self._knot_v,
                "knot_p": None if self._knot_p is None else self._knot_p}

    @classmethod
    def from_state(cls, s: Dict[str, Any]) -> "Representation":
        r = cls(s["kind"], s["center_mode"], s["csi_min"], s["csi_max"],
                {"gauss_knots": s["gauss_knots"], "gauss_p_clip": s["gauss_p_clip"],
                 "logit_margin": s["logit_margin"]})
        r.center = float(s["center"]); r.scale = float(s["scale"])
        r._knot_v = None if s["knot_v"] is None else np.asarray(s["knot_v"])
        r._knot_p = None if s["knot_p"] is None else np.asarray(s["knot_p"])
        return r


def make_representation(cfg: Dict[str, Any], kind: str) -> Representation:
    d = cfg["data"]
    return Representation(kind, d.get("center", "clearsky"),
                          d["csi_min"], d["csi_max"], cfg.get("representation"))


# ============================================================================
