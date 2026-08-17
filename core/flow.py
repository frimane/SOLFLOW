# ===== SECTION: flowmodel ===================================================
# ============================================================================
# Conditional flow matching. A velocity field v(x_t, t | conditioning) is
# trained on the linear path x_t = (1-t) x0 + t x1 with target v = x1 - x0,
# where x0 comes from the anchored prior above and x1 is the real future in
# model space. Generation integrates the ODE from a fresh prior draw.
# The full architecture rationale (why TCN not transformer, why one attention
# bridge, why non-causal convs, FiLM) is in the paper's Appendix A; the
# summary below is the implementation contract.

# The conditioning design:
#   per-position, horizon side : cos(zenith), the valid mask, a forecast-day
#                                index channel, and optionally the clear-sky
#                                GHI profile, all concatenated with x_t as
#                                input channels; and optionally (A) the NWP
#                                forecast channels (DSWRF, TCDC, derived
#                                NWP-CSI) each with a presence sub-mask, so the
#                                net reads the day-ahead weather-model guess per
#                                horizon step and can learn to distrust it where
#                                the presence flag is 0 (filled gap). NWP is a
#                                future-known covariate and appears ONLY here,
#                                never on the history side. The channel set is
#                                frozen at fit time (self.nwp_spec) and restored
#                                on load so the input width always matches the
#                                weights;
#   per-position, history side : a convolutional encoder turns the history
#                                (CSI, cos zenith, mask) into a feature
#                                sequence; the horizon attends to it with
#                                cross-attention, next to one self-attention
#                                layer along the horizon;
#   global                     : a masked mean of the history features plus a
#                                sinusoidal flow-time embedding, injected into
#                                every TCN block through FiLM.
#
# Mask rules show up in exactly three places here: inputs
# are filled with the clear-sky code before the net sees them, the loss
# averages over valid future positions only, and attention layers receive the
# masks as key_padding_masks so padded steps are never used as keys.


# make torch imports local
def _lazy_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _EMAWeights:


    def __init__(self, net, decay):
        torch, _ = _lazy_torch()
        self.decay = float(decay)
        self.step = 0
        self.shadow = {k: v.detach().clone()
                       for k, v in net.state_dict().items()}

    def update(self, net):
        torch, _ = _lazy_torch()
        self.step += 1
        # warmup: early on, trust the current weights more than a shadow still
        # anchored to the random init. The standard min(decay,(1+t)/(10+t))
        # ramp makes EMA robust on short runs instead of being dragged down by
        # the initialization -- without it, a few-epoch fit deploys a barely
        # moved shadow. Converges to `decay` after ~a few hundred steps.
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)

    def swap_in(self, net):

        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(self.shadow)
        return backup


def _make_ema(cfg, net):
    d = float(cfg["train"].get("ema", 0.0) or 0.0)
    return _EMAWeights(net, d) if 0.0 < d < 1.0 else None


def plan_dilations(mcfg: Dict[str, Any], H_out: int) -> Tuple[List[int], int, bool]:

    ks = int(mcfg.get("kernel_size", 3))
    assert ks % 2 == 1, "kernel_size must be odd so padding preserves length"
    max_d = int(mcfg.get("max_dilation", 64))
    max_b = int(mcfg.get("max_blocks", 8))
    n_cfg = mcfg.get("n_blocks", "auto")

    def rf_of(dils):
        return 1 + sum(2 * (ks - 1) * d for d in dils)

    if n_cfg == "auto":
        dils: List[int] = []
        while rf_of(dils) < H_out and len(dils) < max_b:
            dils.append(min(2 ** len(dils), max_d))
    else:
        dils = [min(2 ** i, max_d) for i in range(int(n_cfg))]
    rf = rf_of(dils)
    return dils, rf, rf >= H_out


class _NetBuilder:

    @staticmethod
    def build(cfg, H_in, H_out, n_fut_extra, n_out=1):
        torch, nn = _lazy_torch()
        m = cfg["model"]
        hidden = int(m["hidden"])
        temb = int(m["time_embed_dim"])
        cemb = int(m["cond_embed_dim"])
        drop = float(m["dropout"])
        ks = int(m.get("kernel_size", 3))
        heads = int(m.get("n_heads", 4))
        use_attn = bool(m.get("attention", True))

        dils, rf, covered = plan_dilations(m, H_out)
        if not covered and not use_attn:
            raise ValueError(
                f"TCN receptive field {rf} < horizon {H_out} and attention is "
                f"off; raise model.max_blocks/max_dilation or enable "
                f"model.attention")

        def sinusoidal(t, dim):
            half = dim // 2
            freqs = torch.exp(torch.linspace(0.0, math.log(10000.0), half,
                                             device=t.device))
            ang = t[:, None] * freqs[None, :]
            return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

        class FiLM(nn.Module):
            def __init__(self, ch, cond):
                super().__init__()
                self.g = nn.Linear(cond, ch)
                self.b = nn.Linear(cond, ch)

            def forward(self, x, c):
                return x * (1 + self.g(c)[:, :, None]) + self.b(c)[:, :, None]

        class Block(nn.Module):

            def __init__(self, ch, cond, d):
                super().__init__()
                pad = d * (ks - 1) // 2
                self.c1 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=d)
                self.c2 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=d)
                self.n1 = nn.GroupNorm(1, ch)
                self.n2 = nn.GroupNorm(1, ch)
                self.film = FiLM(ch, cond)
                self.drop = nn.Dropout(drop)

            def forward(self, x, c):
                y = torch.relu(self.n1(self.c1(x)))
                y = self.drop(self.film(y, c))
                y = self.n2(self.c2(y))
                return torch.relu(x + y)

        class AttnBridge(nn.Module):

            def __init__(self, ch):
                super().__init__()
                self.q_pos = nn.Parameter(torch.randn(1, H_out, ch) * 0.02)
                self.m_pos = nn.Parameter(torch.randn(1, H_in, ch) * 0.02)
                self.ln_s = nn.LayerNorm(ch)
                self.ln_q = nn.LayerNorm(ch)
                self.ln_m = nn.LayerNorm(ch)
                self.ln_f = nn.LayerNorm(ch)
                self.self_attn = nn.MultiheadAttention(ch, heads, dropout=drop,
                                                       batch_first=True)
                self.cross_attn = nn.MultiheadAttention(ch, heads, dropout=drop,
                                                        batch_first=True)
                self.ffn = nn.Sequential(nn.Linear(ch, 4 * ch), nn.GELU(),
                                         nn.Linear(4 * ch, ch))

            def forward(self, h, mem, fut_ok, hist_ok):
                # h [B,C,H_out], mem [B,C,H_in], *_ok bool with True = valid
                y = h.transpose(1, 2) + self.q_pos
                z = self.ln_s(y)
                y = y + self.self_attn(z, z, z, key_padding_mask=~fut_ok,
                                       need_weights=False)[0]
                mm = self.ln_m(mem.transpose(1, 2) + self.m_pos)
                # cross-attention into the history memory. A row whose history
                # is FULLY padded (no valid keys) -- which happens for genuine
                # short histories and, deliberately, under history dropout --
                # would make the softmax divide by zero and emit NaN, poisoning
                # the whole batch's gradients. Guard: give such rows a single
                # dummy valid key (its contribution is zeroed out afterwards),
                # so attention returns a finite (zero-information) vector and
                # the model falls back to the FiLM/global conditioning path.
                any_hist = hist_ok.any(dim=1, keepdim=True)      # [B,1]
                kpm = ~hist_ok                                   # True = pad
                safe_kpm = kpm.clone()
                safe_kpm[:, 0] = torch.where(any_hist.squeeze(1),
                                             safe_kpm[:, 0],
                                             torch.zeros_like(safe_kpm[:, 0]))
                ca = self.cross_attn(self.ln_q(y), mm, mm,
                                     key_padding_mask=safe_kpm,
                                     need_weights=False)[0]
                ca = ca * any_hist.float().unsqueeze(-1)         # zero the dummy
                y = y + ca
                y = y + self.ffn(self.ln_f(y))
                return y.transpose(1, 2)

        class VelocityNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Conv1d(1 + n_fut_extra, hidden, 1)
                self.hist_enc = nn.Sequential(
                    nn.Conv1d(3, hidden, 5, padding=2), nn.GELU(),
                    nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
                self.hist_proj = nn.Linear(hidden, cemb)
                self.time_mlp = nn.Sequential(
                    nn.Linear(temb, cemb), nn.SiLU(), nn.Linear(cemb, cemb))
                self.blocks = nn.ModuleList(
                    [Block(hidden, cemb, d) for d in dils])
                self.bridge = AttnBridge(hidden) if use_attn else None
                # insert the bridge midway so it sees TCN features and its
                # output is refined by the remaining blocks
                self.bridge_after = max(0, len(dils) // 2 - 1)
                self.out = nn.Conv1d(hidden, n_out, 1)
                self.receptive_field = rf
                self._temb = temb
                self._n_out = n_out

            def forward(self, x, t, hist_feats, hist_ok, fut_feats, fut_ok):
                # x [B,H_out]; fut_feats [B,n_extra,H_out];
                # hist_feats [B,3,H_in]; *_ok bool masks with True = valid
                h = self.in_proj(torch.cat([x.unsqueeze(1), fut_feats], dim=1))
                mem = self.hist_enc(hist_feats)
                w = hist_ok.float().unsqueeze(1)
                g = (mem * w).sum(-1) / w.sum(-1).clamp_min(1.0)
                cond = self.time_mlp(sinusoidal(t, self._temb)) \
                    + self.hist_proj(g)
                for i, blk in enumerate(self.blocks):
                    h = blk(h, cond)
                    if self.bridge is not None and i == self.bridge_after:
                        h = self.bridge(h, mem, fut_ok, hist_ok)
                out = self.out(h)
                return out.squeeze(1) if self._n_out == 1 else out

        return VelocityNet()


class FlowMatcher:


    def __init__(self, cfg, H_in, H_out, K, n_days, prior, rep, device):
        self.cfg = cfg
        self.H_in = int(H_in)
        self.H_out = int(H_out)
        self.K = int(K)
        self.n_days = int(n_days)
        self.prior = prior
        self.rep = rep
        self.device = device
        self.net = None
        # padded inputs are filled with the clear-sky code: finite and
        # in-distribution under every representation/centering combination
        self.pad_fill = float(rep.clearsky_code())
        self.gcs_scale: Optional[float] = None
        # (A) NWP horizon conditioning. `nwp_spec` is the ordered list of
        # (array_key, kind) the net was built with; frozen at fit time and
        # restored on load so the channel count always matches the weights.
        # kind in {"ghi_norm","unit","csi","present"} selects normalization.
        self.nwp_spec: List[Tuple[str, str]] = []
        # static site features (normalized lat/lon/alt) as constant horizon
        # channels; frozen at fit time so the channel count is reproducible on
        # load. None -> site conditioning off. Length is fixed at 3.
        self.site_vec: Optional[np.ndarray] = None
        # optional post-hoc EMOS-style member calibrator (fit on validation,
        # applied at prediction). None until fit_calibrator() is called.
        self.calibrator = None

    def _resolve_site_vec(self):

        if not bool(self.cfg["model"].get("condition_site", False)):
            return None
        if self.cfg.get("sites"):
            # multi-site: placeholder = mean coordinate across the pool
            vecs = [_site_coord_vec(self.cfg, s["latitude"], s["longitude"],
                                    s["altitude"]) for s in self.cfg["sites"]]
            return np.mean(vecs, axis=0).astype(np.float32) if vecs else None
        s = self.cfg.get("site")
        if not s:
            return None
        return _site_coord_vec(self.cfg, s["latitude"], s["longitude"],
                               s["altitude"])

    # ---- small helpers -------------------------------------------------------
    def _to_t(self, a):
        torch, _ = _lazy_torch()
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32,
                               device=self.device)

    def _to_b(self, a):
        torch, _ = _lazy_torch()
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.bool,
                               device=self.device)

    def _fill(self, x_np, mask_np):
        return np.where(mask_np, np.nan_to_num(x_np, nan=self.pad_fill),
                        self.pad_fill).astype(np.float32)

    @staticmethod
    def _cos_zen(zen):
        return np.cos(np.deg2rad(np.nan_to_num(zen, nan=90.0))).astype(np.float32)

    def _dayfrac_row(self):
        f = (np.arange(self.n_days, dtype=np.float32) + 0.5) / self.n_days
        return np.repeat(f, self.K)                              # [H_out]

    def _fut_extras(self, fut_zen, fut_mask, fut_ghi_cs, fut_nwp=None,
                    site_coords=None):
        N = fut_zen.shape[0]
        ch = [self._cos_zen(fut_zen),
              fut_mask.astype(np.float32),
              np.broadcast_to(self._dayfrac_row(), (N, self.H_out)).copy()]
        if self.gcs_scale is not None:
            if fut_ghi_cs is None:
                g = np.zeros((N, self.H_out), np.float32)
            else:
                g = np.nan_to_num(np.asarray(fut_ghi_cs, np.float32)
                                  / self.gcs_scale, nan=0.0)
            ch.append(g)
        for key, kind in self.nwp_spec:
            arr = None if fut_nwp is None else fut_nwp.get(key)
            if arr is None:
                ch.append(np.zeros((N, self.H_out), np.float32))
                continue
            a = np.asarray(arr)
            if kind == "present":
                ch.append(a.astype(np.float32))
            elif kind == "ghi_norm":
                sc = self.gcs_scale if self.gcs_scale else 1000.0
                ch.append(np.nan_to_num(a.astype(np.float32) / sc, nan=0.0))
            elif kind == "unit":                       # percent -> [0,1]
                ch.append(np.nan_to_num(a.astype(np.float32) / 100.0, nan=0.0))
            else:                                       # "csi": already a ratio
                ch.append(np.nan_to_num(a.astype(np.float32), nan=0.0))
        # static site features: each a constant channel broadcast across the
        # horizon. Constant over steps, so they identify the SITE without
        # varying within a day; the net reads them like any other channel.
        # Multi-site: per-window site_coords [N,3] give each window ITS OWN
        # station's coordinates. Single-site: the frozen self.site_vec is
        # broadcast to all windows. The presence of self.site_vec (set at fit
        # time) decides whether these channels exist at all, so the channel
        # count is fixed and matches the weights on load either way.
        if self.site_vec is not None:
            n_site = int(self.site_vec.shape[0])
            if site_coords is not None:
                sc = np.asarray(site_coords, np.float32)
                if sc.shape != (N, n_site):
                    raise ValueError(
                        f"site_coords shape {sc.shape} != expected {(N, n_site)}")
                for j in range(n_site):
                    ch.append(np.repeat(sc[:, j:j + 1], self.H_out, axis=1))
            else:
                for val in self.site_vec:
                    ch.append(np.full((N, self.H_out), float(val), np.float32))
        return np.stack(ch, axis=1).astype(np.float32)

    @property
    def _n_fut_extra(self):
        return (3 + (1 if self.gcs_scale is not None else 0)
                + len(self.nwp_spec)
                + (0 if self.site_vec is None else int(self.site_vec.shape[0])))

    @staticmethod
    def _nwp_kind_for(key):
        if key.endswith("_present"):
            return "present"
        if "csi" in key:
            return "csi"
        if "tcdc" in key or "tcc" in key:
            return "unit"
        if "dswrf" in key or "dswrf" in key or "ghi" in key or "swrf" in key:
            return "ghi_norm"
        return "csi"

    def _nwp_anchor_array(self, fut_nwp, fut_mask):


        if not fut_nwp:
            return None
        arr = fut_nwp.get("fut_nwp_csi")
        if arr is None:
            return None
        enc = self.rep.encode(np.asarray(arr, np.float64))
        return self._fill(enc, np.asarray(fut_mask, bool))

    def _resolve_nwp_spec(self, W):

        if not bool(self.cfg["model"].get("condition_nwp", False)):
            return []
        nc = self.cfg["data"].get("nwp") or {}
        wanted = list(nc.get("channels", []))
        if nc.get("derive_csi", False):
            wanted = wanted + ["nwp_csi"]
        spec = []
        for base in wanted:
            key = f"fut_{base}"
            if key in W:
                spec.append((key, self._nwp_kind_for(base)))
                pk = f"fut_{base}_present"
                if pk in W:
                    spec.append((pk, "present"))
        return spec

    # ---- ODE integration ------------------------------------------------------
    def _integrate(self, x, hist_t, hist_ok, fut_t, fut_ok, steps):
        torch, _ = _lazy_torch()
        sampler = str(self.cfg["train"].get("sampler", "heun")).lower()
        dt = 1.0 / steps
        B = x.shape[0]

        def vel(state, ts):
            t = torch.full((B,), ts, device=self.device, dtype=state.dtype)
            return self.net(state, t, hist_t, hist_ok, fut_t, fut_ok)

        for s in range(steps):
            t0 = s * dt
            if sampler == "euler":
                x = x + dt * vel(x, t0)
            elif sampler == "midpoint":
                k1 = vel(x, t0)
                x = x + dt * vel(x + 0.5 * dt * k1, t0 + 0.5 * dt)
            elif sampler == "heun":
                k1 = vel(x, t0)
                k2 = vel(x + dt * k1, t0 + dt)
                x = x + 0.5 * dt * (k1 + k2)
            else:
                raise ValueError(f"unknown sampler '{sampler}'")
        return x

    # ---- training ---------------------------------------------------------------
    def fit(self, hist_csi, fut_csi, hist_zen, fut_zen, hist_mask, fut_mask,
            fut_ghi_cs=None, fut_nwp=None, es_split=None, rng=None,
            site_coords=None, warm_start=False):

        torch, nn = _lazy_torch()
        t_cfg = self.cfg["train"]
        rng = rng or np.random.default_rng(self.cfg["seed"])

        # Preserve the checkpoint's representation, conditioning width, and
        # clear-sky normalization during warm-start fine-tuning. Re-estimating
        # these from a new dataset would change the input contract silently.
        if warm_start:
            if self.net is None:
                raise ValueError("warm_start=True requires a loaded checkpoint")
            expected_spec = self._resolve_nwp_spec(fut_nwp or {})
            if expected_spec != self.nwp_spec:
                raise ValueError(
                    "new dataset NWP channel contract does not match the checkpoint: "
                    f"checkpoint={self.nwp_spec}, dataset={expected_spec}")
            if bool(self.cfg["model"].get("condition_clearsky_ghi", True)) \
                    and fut_ghi_cs is not None and self.gcs_scale is None:
                raise ValueError("checkpoint has no clear-sky GHI scale but the "
                                 "new dataset provides clear-sky conditioning")
            if self.site_vec is not None and site_coords is None and self.cfg.get("sites"):
                raise ValueError("site-conditioned checkpoint requires site_coords "
                                 "for the new dataset")
        else:
            # clear-sky GHI conditioning: fix the normalization on this fit's data
            if bool(self.cfg["model"].get("condition_clearsky_ghi", True)) \
                    and fut_ghi_cs is not None:
                vals = np.asarray(fut_ghi_cs, np.float64)[np.asarray(fut_mask, bool)]
                vals = vals[np.isfinite(vals)]
                vals = vals[np.isfinite(vals)]
                self.gcs_scale = max(1.0, float(np.percentile(vals, 99))) \
                    if vals.size else None
            else:
                self.gcs_scale = None
            # (A) freeze the NWP channel spec from the keys actually present
            self.nwp_spec = self._resolve_nwp_spec(fut_nwp or {})
            # freeze the static site vector (normalized lat/lon/alt) if enabled
            self.site_vec = self._resolve_site_vec()
            self.net = _NetBuilder.build(self.cfg, self.H_in, self.H_out,
                                         self._n_fut_extra).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=t_cfg["lr"],
                                weight_decay=t_cfg["weight_decay"])

        x1 = self.rep.encode(fut_csi)                      # NaN at padding
        hc = self.rep.encode(hist_csi)
        hz = self._cos_zen(hist_zen)
        fut_ex = self._fut_extras(fut_zen, fut_mask, fut_ghi_cs, fut_nwp,
                                  site_coords=site_coords)
        hm = np.asarray(hist_mask, bool)
        fm = np.asarray(fut_mask, bool)
        hist_csi = np.asarray(hist_csi, np.float32)

        # (B) NWP-CSI anchor in model space for the 'nwp' prior. Encode the
        # physical NWP-CSI forecast through the SAME representation as the
        # target, fill padding with the clear-sky code. None when unavailable.
        nwp_anchor = self._nwp_anchor_array(fut_nwp, fm)
        if getattr(self.prior, "needs_nwp", False) and nwp_anchor is None:
            warnings.warn("prior 'nwp' selected but no NWP-CSI channel present; "
                          "anchoring on clear sky instead", RuntimeWarning)

        N = x1.shape[0]
        if es_split is not None and t_cfg.get("early_stopping", True):
            tr_idx, va_idx = (np.asarray(es_split[0]), np.asarray(es_split[1]))
            do_es = len(va_idx) > 0
        else:
            tr_idx, va_idx, do_es = np.arange(N), None, False

        # ---- move the static training tensors to the device ONCE -----------
        # The per-batch cost then reduces to: index on-device + ship the prior
        # draw (numpy by design, so the prior stays testable without torch).
        # Without this, every batch performs ~6 synchronous host->device
        # copies and the GPU starves on transfers. Numpy copies of the filled
        # history are kept because prior.sample consumes them on the CPU.
        # If the allocation fails (very large N on a small GPU) we fall back
        # to the original per-batch-transfer path.
        hcf_np = self._fill(hc, hm)                       # numpy, for the prior
        x1f_np = self._fill(x1, fm)
        hist_feats_np = np.stack([hcf_np, hz, hm.astype(np.float32)], axis=1)
        on_device = False
        try:
            X1 = self._to_t(x1f_np)                       # [N,H]
            HF = self._to_t(hist_feats_np)                # [N,3,H_in]
            HM = self._to_b(hm)                           # [N,H_in] bool
            FX = self._to_t(fut_ex)                       # [N,C,H]
            FB = self._to_b(fm)                           # [N,H] bool
            FF = self._to_t(fm.astype(np.float32))        # [N,H] float
            on_device = True
        except RuntimeError as e:                         # e.g. CUDA OOM
            warnings.warn(f"device cache of training tensors failed ({e}); "
                          f"falling back to per-batch transfers", RuntimeWarning)

        # optional bf16 autocast (train.amp: 'off' | 'bf16'); no GradScaler is
        # needed for bf16. Default off so published-run numerics are unchanged.
        use_amp = (str(t_cfg.get("amp", "off")).lower() == "bf16"
                   and str(self.device).startswith("cuda"))

        best = np.inf
        best_state = None
        bad = 0
        bs = int(t_cfg["batch_size"])
        ema = _make_ema(self.cfg, self.net)   # None when train.ema disabled
        for ep in range(int(t_cfg["epochs"])):
            self.net.train()
            perm = rng.permutation(tr_idx)
            # accumulate the running loss ON the device and synchronize once
            # per epoch; float(loss) every batch would stall the pipeline
            running_t = torch.zeros((), device=self.device)
            nb = 0
            for s0 in range(0, len(perm), bs):
                b = perm[s0:s0 + bs]
                hmb = hm[b]
                hcb = hcf_np[b]
                x0 = self.prior.sample(len(b), rng, hist_rep=hcb,
                                       hist_mask=hmb, fut_mask=fm[b],
                                       hist_csi_phys=hist_csi[b],
                                       fut_nwp_anchor=(None if nwp_anchor is None
                                                       else nwp_anchor[b]))
                tt = rng.random(len(b)).astype(np.float32)
                x0_t = self._to_t(x0)
                tt_t = self._to_t(tt)
                if on_device:
                    bt = torch.as_tensor(b, device=self.device)
                    x1b_t = X1[bt]
                    xt_t = (1 - tt_t)[:, None] * x0_t + tt_t[:, None] * x1b_t
                    target_t = x1b_t - x0_t
                    hist_t, hm_t = HF[bt], HM[bt]
                    fx_t, fb_t, fm_t = FX[bt], FB[bt], FF[bt]
                else:
                    x1b_t = self._to_t(x1f_np[b])
                    xt_t = (1 - tt_t)[:, None] * x0_t + tt_t[:, None] * x1b_t
                    target_t = x1b_t - x0_t
                    hist_t = self._to_t(hist_feats_np[b])
                    hm_t = self._to_b(hmb)
                    fx_t = self._to_t(fut_ex[b])
                    fb_t = self._to_b(fm[b])
                    fm_t = self._to_t(fm[b].astype(np.float32))

                # history (conditioning) dropout: blank the history of a random
                # subset of rows so the model learns to forecast from NWP +
                # geometry alone. We force the history mask all-False (the
                # cross-attention dummy-key guard + masked-mean clamp keep this
                # NaN-safe) AND zero the history feature content so no signal
                # leaks through the conv encoder. The prior draw x0 is left as
                # is: dropout trains the VELOCITY FIELD to cope without history;
                # the anchor is a separate, deployment-time fallback.
                p_hd = float(t_cfg.get("history_dropout", 0.0) or 0.0)
                if p_hd > 0.0:
                    drop = self._to_b(rng.random(len(b)) < p_hd)  # [B] bool
                    if drop.any():
                        keep = (~drop).float()
                        hist_t = hist_t * keep[:, None, None]
                        hm_t = hm_t & (~drop)[:, None]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_amp):
                    v = self.net(xt_t, tt_t, hist_t, hm_t, fx_t, fb_t)
                    diff2 = (v.float() - target_t) ** 2 * fm_t
                    loss = diff2.sum() / fm_t.sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update(self.net)
                running_t += loss.detach()
                nb += 1
            running = float(running_t)                    # one sync per epoch
            if t_cfg.get("verbose", 1) >= 1:
                print(f"  epoch {ep+1}/{t_cfg['epochs']} "
                      f"loss={running/max(nb,1):.5f}", flush=True)

            # early stopping on fair, mask-aware validation CRPS at
            # deployment-like sampling fidelity. Evaluated with the EMA
            # weights (the weights we would actually deploy), and best_state
            # snapshots THOSE, so selection and deployment see the same model.
            if do_es and (ep % int(t_cfg.get("es_every", 2)) == 0):
                sel = va_idx[:int(t_cfg.get("es_max_rows", 512))]
                raw_backup = ema.swap_in(self.net) if ema is not None else None
                pred = self._sample_rows(
                    sel, hc, hz, hist_csi, hm, fut_ex, fm,
                    M=int(t_cfg.get("es_members", 50)),
                    steps=int(t_cfg.get("es_sampling_steps", 20)), rng=rng,
                    nwp_anchor=nwp_anchor)
                vc = crps_masked(pred, np.asarray(fut_csi)[sel], fm[sel],
                                 fair=True)
                if vc < best - float(t_cfg.get("min_delta", 1e-4)):
                    best, bad = vc, 0
                    if t_cfg.get("restore_best", True):
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                if raw_backup is not None:
                    self.net.load_state_dict(raw_backup)   # resume training
                if bad >= int(t_cfg.get("patience", 15)):
                    if t_cfg.get("verbose", 1) >= 1:
                        print(f"  early stop at epoch {ep+1} "
                              f"(val fair CRPS {best:.5f})")
                    break
        # finalize: best EMA snapshot if ES picked one, else the EMA weights,
        # else the last raw weights
        if best_state is not None:
            self.net.load_state_dict(best_state)
        elif ema is not None:
            ema.swap_in(self.net)
        return self

    # ---- sampling ---------------------------------------------------------------
    def _sample_rows(self, sel, hc, hz, hist_csi_phys, hm, fut_ex, fm,
                     M, steps, rng, nwp_anchor=None):

        torch, _ = _lazy_torch()
        sel = np.asarray(sel)
        chunk_rows = max(1, int(self.cfg["train"].get("sample_chunk", 4096)) // M)
        out = np.empty((len(sel), M, self.H_out), np.float32)
        self.net.eval()
        with torch.no_grad():
            for c0 in range(0, len(sel), chunk_rows):
                rows = sel[c0:c0 + chunk_rows]
                rep_ = lambda a: np.repeat(a, M, axis=0)
                hcb = self._fill(hc[rows], hm[rows])
                na = (None if nwp_anchor is None
                      else rep_(np.asarray(nwp_anchor)[rows]))
                x0 = self.prior.sample(
                    len(rows) * M, rng,
                    hist_rep=rep_(hcb), hist_mask=rep_(hm[rows]),
                    fut_mask=rep_(fm[rows]),
                    hist_csi_phys=rep_(np.asarray(hist_csi_phys)[rows]),
                    fut_nwp_anchor=na)
                # ship each conditioning tensor ONCE per chunk and repeat it
                # M-fold on the device: repeating on the CPU first would move
                # M x more bytes across the bus for identical content. Only x0
                # crosses at full size -- it is the randomness itself.
                rt = lambda t: t.repeat_interleave(M, dim=0)
                hist_t = rt(self._to_t(np.stack(
                    [hcb, hz[rows], hm[rows].astype(np.float32)], axis=1)))
                x = self._integrate(self._to_t(x0), hist_t,
                                    rt(self._to_b(hm[rows])),
                                    rt(self._to_t(fut_ex[rows])),
                                    rt(self._to_b(fm[rows])), steps)
                z = x.cpu().numpy().reshape(len(rows), M, self.H_out)
                out[c0:c0 + len(rows)] = self.rep.decode(z)
        return out

    def predict_ensemble(self, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         fut_ghi_cs=None, fut_nwp=None, n_ensemble=50, rng=None,
                         raw=False, site_coords=None):

        rng = rng or np.random.default_rng(0)
        hc = self.rep.encode(hist_csi)
        hz = self._cos_zen(hist_zen)
        fmb = np.asarray(fut_mask, bool)
        fut_ex = self._fut_extras(np.asarray(fut_zen), fmb, fut_ghi_cs, fut_nwp,
                                  site_coords=site_coords)
        nwp_anchor = self._nwp_anchor_array(fut_nwp, fmb)
        ens = self._sample_rows(
            np.arange(np.asarray(hist_csi).shape[0]), hc, hz,
            np.asarray(hist_csi, np.float32), np.asarray(hist_mask, bool),
            fut_ex, fmb,
            M=int(n_ensemble),
            steps=int(self.cfg["train"]["n_sampling_steps"]), rng=rng,
            nwp_anchor=nwp_anchor)
        if self.calibrator is not None and not raw:
            ens = self.calibrator.apply(ens)
        return ens

    def fit_calibrator(self, W, va_idx, rng=None):

        if not bool(self.cfg["train"].get("calibrate", False)):
            return self
        va_idx = np.asarray(va_idx)
        if len(va_idx) < 20:
            return self
        rng = rng or np.random.default_rng(self.cfg["seed"])
        have_gcs = "fut_ghi_cs" in W
        nwp_keys = [k for k in W if k.startswith("fut_")
                    and k not in ("fut_csi", "fut_zen", "fut_mask", "fut_ghi_cs")]
        fut_nwp = None if not nwp_keys else {k: W[k][va_idx] for k in nwp_keys}
        sc = W["site_coords"][va_idx] if "site_coords" in W else None
        raw_ens = self.predict_ensemble(
            W["hist_csi"][va_idx], W["hist_zen"][va_idx], W["fut_zen"][va_idx],
            W["hist_mask"][va_idx], W["fut_mask"][va_idx],
            fut_ghi_cs=(W["fut_ghi_cs"][va_idx] if have_gcs else None),
            fut_nwp=fut_nwp, site_coords=sc,
            n_ensemble=int(self.cfg["experiment"]["n_ensemble"]),
            rng=rng, raw=True)
        cal = MemberCalibrator(self.K, self.n_days).fit(
            raw_ens, W["fut_csi"][va_idx], W["fut_mask"][va_idx])
        self.calibrator = cal
        return self

    # ---- persistence ---------------------------------------------------------
    def save(self, path):

        torch, _ = _lazy_torch()
        ckpt = {
            "format": "day_ahead_flow_v2",
            "cfg": self.cfg,
            "H_in": self.H_in, "H_out": self.H_out,
            "K": self.K, "n_days": self.n_days,
            "pad_fill": self.pad_fill, "gcs_scale": self.gcs_scale,
            "nwp_spec": self.nwp_spec,
            "site_vec": (None if self.site_vec is None
                         else self.site_vec.tolist()),
            "calibrator": (None if self.calibrator is None
                           else self.calibrator.state()),
            "net_state": self.net.state_dict(),
            "rep": self.rep.state(),
            "prior": self.prior.state(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ckpt, path)
        return path

    @classmethod
    def load(cls, path, device=None):
        torch, _ = _lazy_torch()
        device = device or get_device("auto")
        # weights_only=False because the checkpoint holds numpy arrays and a
        # config dict; only load checkpoints from your own runs
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get("format") != "day_ahead_flow_v2":
            raise ValueError(f"unrecognized checkpoint format in {path}")
        cfg = ckpt["cfg"]
        rep = Representation.from_state(ckpt["rep"])
        prior = DayPrior.from_state(ckpt["prior"], cfg)
        fm = cls(cfg, ckpt["H_in"], ckpt["H_out"], ckpt["K"], ckpt["n_days"],
                 prior, rep, device)
        fm.pad_fill = float(ckpt["pad_fill"])
        fm.gcs_scale = ckpt.get("gcs_scale")
        # restore the NWP channel spec BEFORE building the net so its input
        # width matches the saved weights (tuples survive torch.save as lists)
        fm.nwp_spec = [tuple(x) for x in ckpt.get("nwp_spec", [])]
        sv = ckpt.get("site_vec")
        fm.site_vec = None if sv is None else np.asarray(sv, np.float32)
        fm.calibrator = MemberCalibrator.from_state(ckpt.get("calibrator"))
        fm.net = _NetBuilder.build(cfg, fm.H_in, fm.H_out,
                                   fm._n_fut_extra).to(device)
        fm.net.load_state_dict(ckpt["net_state"])
        fm.net.eval()
        return fm


# ============================================================================
