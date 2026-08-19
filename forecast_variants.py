"""
================================================================================
 forecast_variants.py  --  forecast every variant, plot them against each other
================================================================================

Superset of the old forecast_and_plot.py. Same job (load a checkpoint, forecast,
draw a fan chart) plus everything needed to stress-test a model's behaviour
when its inputs are incomplete, or absent entirely:

  1. AVAILABILITY VARIANTS -- forecast the SAME rows under up to four
     information conditions: full, no-history, no-NWP, neither. Each variant
     is only computed if the checkpoint's OWN training config/prior shows it
     was actually prepared for that condition (capability_report()) --
     otherwise it is SKIPPED by default with a printed reason, not silently
     forecast as if it were trustworthy. Pass --allow-unsupported to compute
     a skipped variant anyway, loudly labeled as a diagnostic. Concretely:
     'no_history' needs train.history_dropout > 0 (this codebase's only
     dropout knob). 'no_nwp' needs EITHER an NWP-dropout regularizer (does
     not exist in this codebase yet) OR a prior that isn't NWP-anchored --
     NWP enters this codebase two separate ways (conditioning channels vs
     the 'nwp' prior's anchor) and only the latter (prior.kind == 'nwp')
     actually depends on NWP being present; the conditioning channels
     zero-fill safely by architecture regardless of prior. See the
     SAFEGUARDS note below.

  2. COLD-START GENERATION -- forecast for a bare location (lat/lon/alt) and a
     calendar date range with NO sensor history and NO NWP at all: the only
     inputs are solar geometry (zenith) and a clear-sky irradiance model
     (pvlib Ineichen), computed from scratch. This is the "new site, day one"
     case -- what the model can say about a place before any data exists.
     Same capability guard as above: refuses by default unless the
     checkpoint was trained to cope with missing history AND NWP absence is
     safe for it (no NWP conditioning at all, NWP-dropout training, or a
     prior that isn't NWP-anchored).

  3. ROLLOUT VARIANTS -- chain the forecast forward in blocks of
     `task.forecast_days` days two ways:
       - "direct"        : every block is forecast from its own TRUE recorded
                            history (teacher forcing / no compounding).
       - "autoregressive" : only the FIRST block uses true history; every
                            later block's history is the model's OWN previous
                            forecast fed back in (point feedback by default --
                            the ensemble median -- or full path-wise ensemble
                            feedback with --rollout-ensemble).
     Comparing the two shows how fast the forecast degrades under compounding
     error vs a model that is always re-grounded in real observations.

  4. MULTI-MODEL COMPARISON -- overlay a chosen flow-matching checkpoint
     against the deep_quantile baseline and/or the classical baselines
     (day_persistence, peen, ch_peen, analog_day, nwp_direct) on one fan
     chart: the flow model gets full quantile shading, everything else is
     drawn as a single median/point line for reference. Shape-checked up
     front (H_in/H_out/K) so a mismatched deep_quantile checkpoint fails
     fast with a clear message instead of a cryptic tensor error, and any
     baseline that can't fit on this windows file (e.g. nwp_direct with no
     fut_nwp_csi) is skipped with a reason rather than crashing the run.

SAFEGUARDS -- "what a checkpoint doesn't have, it can't do"
  capability_report(fm) reads a checkpoint's OWN training config/prior
  (never its output) to decide what it was actually prepared for: whether
  train.history_dropout was on, whether it has NWP conditioning at all,
  whether its PRIOR is NWP-anchored (prior.kind == 'nwp', via
  DayPrior.needs_nwp -- distinct from having NWP conditioning channels), and
  (today, always False, since no such knob exists in this codebase) whether
  it had NWP-dropout training. availability_variants() and
  coldstart_forecast() consult this before computing anything, and refuse
  (skip, or raise UnsupportedByCheckpoint) rather than hand back a forecast
  for a condition the network was never exposed to -- EXCEPT for 'no_nwp' on
  a non-NWP-anchored-prior checkpoint, which is allowed by default because
  fullCode_v2.py's own conditioning code (FlowMatcher._fut_extras) zero-
  fills absent NWP channels architecturally, not as a guess. Every guard has
  one explicit override, --allow-unsupported, which still runs the request
  but labels the result as a diagnostic probe of off-manifold behavior, not
  a forecast -- and _warn_if_ood() double-checks the output afterward in
  case it still looks physically implausible.

NOTE on "a lot of dropout logic" -- there are two, different, dropout knobs:
  * train.history_dropout (model.py): a TRAINING regularizer that blanks
    history for a random subset of BATCHES so the velocity field learns to
    cope without it. It has no effect at inference time by itself.
  * inference-time "missing" inputs (this file): hist_mask=all-False /
    fut_nwp=None. The network was built to be NaN/absence-safe for BOTH
    (masked-mean clamp + cross-attention dummy-key guard + zero-filled NWP
    channels), so passing genuinely absent data at forecast time is exactly
    the situation training tried to prepare it for -- this file's job is to
    make trying that easy and to test that it doesn't crash.
  Availability variant "no_history" additionally zeroes the HISTORY ZENITH
  channel (not just content+mask), because that is what history_dropout
  actually zeroed during training (see FlowMatcher.fit); leaving zenith
  "leaking" through would test a condition the model never saw.

--------------------------------------------------------------------------------
 USAGE
--------------------------------------------------------------------------------
  # what checkpoints do I have?
  python forecast_variants.py list --models-dir models/day_ahead

  # plain forecast + plot (same as the old forecast_and_plot.py)
  python forecast_variants.py forecast --checkpoint models/day_ahead/pooled_flow_gauss_nwp_final.pt \
      --windows data/day_ahead_windows.npz --rows 0 1 --mode ghi --out forecasts

  # what does the model say with/without history, with/without NWP?
  python forecast_variants.py availability --checkpoint ...final.pt \
      --windows data/day_ahead_windows.npz --rows 0 --out forecasts

  # cold start: nothing but a location and the calendar
  python forecast_variants.py coldstart --checkpoint ...final.pt \
      --lat 40.05192 --lon -88.37309 --alt 213 --start-date 2024-06-01 --out forecasts

  # autoregressive vs direct 9-day rollout (3 blocks of 3 days), starting at row 0
  python forecast_variants.py rollout --checkpoint ...final.pt \
      --windows data/day_ahead_windows.npz --start-row 0 --n-blocks 3 --out forecasts

  # flow model vs deep_quantile vs every classical baseline, one chart
  python forecast_variants.py compare --checkpoint ...final.pt \
      --deep-quantile-checkpoint ...deep_quantile_final.pt \
      --windows data/day_ahead_windows.npz --rows 0 --baselines all --out forecasts

  # exercise every code path above against synthetic data (no files needed)
  python forecast_variants.py selftest
================================================================================
"""
from __future__ import annotations
import os
import re
import glob
import argparse
import math
import warnings
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# the main module: training / FlowMatcher / DeepQuantile / baselines / reps /
# priors. Try the "installed as a package" layout first (companion-file
# convention used by the original forecast_and_plot.py), then fall back to a
# flat sibling file so this script also works dropped next to fullCode_v2.py.
# try:
#     import src.flowmodel as core
# except ImportError:                                     # pragma: no cover
import core


# ============================================================================
# ===== SECTION: checkpoint discovery ========================================
# ============================================================================
_FLOW_RE = re.compile(
    r"^(?:(?P<run_tag>.+)_)?flow_(?P<rep>[a-zA-Z0-9]+)_(?P<prior>[a-zA-Z0-9]+)_"
    r"(?:fold(?P<fold>\d+)|(?P<final>final))\.pt$")
_DQ_RE = re.compile(
    r"^(?:(?P<run_tag>.+)_)?deep_quantile_"
    r"(?:fold(?P<fold>\d+)|(?P<final>final))\.pt$")


def discover_checkpoints(models_dir: str) -> List[Dict[str, Any]]:
    """Scan `models_dir` for every flow_* / deep_quantile_* checkpoint this
    codebase could have written and parse its (kind, rep, prior, fold/final,
    run_tag) out of the filename. Unrecognized .pt files are listed with
    kind='unknown' rather than dropped, so nothing silently disappears."""
    out = []
    for path in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
        name = os.path.basename(path)
        m = _FLOW_RE.match(name)
        if m:
            g = m.groupdict()
            out.append({"path": path, "kind": "flow",
                        "run_tag": g["run_tag"] or "", "rep": g["rep"],
                        "prior": g["prior"],
                        "fold": (None if g["final"] else int(g["fold"]))})
            continue
        m = _DQ_RE.match(name)
        if m:
            g = m.groupdict()
            out.append({"path": path, "kind": "deep_quantile",
                        "run_tag": g["run_tag"] or "", "rep": None,
                        "prior": None,
                        "fold": (None if g["final"] else int(g["fold"]))})
            continue
        out.append({"path": path, "kind": "unknown", "run_tag": None,
                    "rep": None, "prior": None, "fold": None})
    return out


def print_checkpoints(models_dir: str):
    ckpts = discover_checkpoints(models_dir)
    if not ckpts:
        print(f"no .pt checkpoints found under {models_dir}")
        return
    for c in ckpts:
        tag = "final" if c["fold"] is None else f"fold{c['fold']}"
        if c["kind"] == "flow":
            print(f"  [flow]           rep={c['rep']:<7} prior={c['prior']:<12} "
                  f"{tag:<7} run_tag={c['run_tag'] or '-':<12} {c['path']}")
        elif c["kind"] == "deep_quantile":
            print(f"  [deep_quantile]  {tag:<7} "
                  f"run_tag={c['run_tag'] or '-':<12} {c['path']}")
        else:
            print(f"  [unknown]        {c['path']}")


def load_any(path: str, device=None):
    """Load a checkpoint regardless of whether it is a FlowMatcher or a
    DeepQuantile save -- reads the `format` tag inside and dispatches."""
    import torch
    device = device or core.get_device("auto")
    peek = torch.load(path, map_location=device, weights_only=False)
    fmt = peek.get("format")
    if fmt == "day_ahead_flow_v2":
        return core.FlowMatcher.load(path, device=device)
    if fmt == "deep_quantile_v1":
        return core.DeepQuantile.load(path, device=device)
    raise ValueError(f"{path}: unrecognized checkpoint format {fmt!r}")


BASELINE_CLASSES = {
    "day_persistence": lambda K, Fd, cfg: core.DayPersistence(K, Fd),
    "peen": lambda K, Fd, cfg: core.PeEn(
        K, Fd, cfg["experiment"].get("peen_days", 3)),
    "ch_peen": lambda K, Fd, cfg: core.CHPeEn(K, Fd),
    "analog_day": lambda K, Fd, cfg: core.AnalogDay(K, Fd),
    "blend": lambda K, Fd, cfg: core.BlendPersistence(
        K, Fd, mode="least_squares"),
    "blend_corr": lambda K, Fd, cfg: core.BlendPersistence(
        K, Fd, mode="correlation"),
    "nwp_direct": lambda K, Fd, cfg: core.NWPDirect(K, Fd),
}


def fit_baselines(names: List[str], W: Dict[str, np.ndarray], cfg: dict,
                  K: int, n_days: int, fit_idx: Optional[np.ndarray] = None):
    """Fit the requested classical baselines on windows `fit_idx` (default:
    all rows in W). Baselines aren't checkpointed by `run` (they're cheap
    numpy objects, refit on demand), so this is how a comparison plot gets
    one. Rows that can't be fit (e.g. nwp_direct with no NWP in W) are
    skipped with a warning rather than raising, so one bad baseline doesn't
    sink the whole comparison."""
    fit_idx = np.arange(W["fut_csi"].shape[0]) if fit_idx is None \
        else np.asarray(fit_idx)
    fitted = {}
    for name in names:
        if name not in BASELINE_CLASSES:
            warnings.warn(f"unknown baseline '{name}', skipping")
            continue
        try:
            b = BASELINE_CLASSES[name](K, n_days, cfg)
            if name == "analog_day":
                b.fit(W["hist_csi"][fit_idx], W["fut_csi"][fit_idx],
                     W["hist_mask"][fit_idx], W["fut_mask"][fit_idx])
            elif name in {"blend", "blend_corr"}:
                b.fit(W["hist_csi"][fit_idx], W["fut_csi"][fit_idx],
                     W["hist_mask"][fit_idx], W["fut_mask"][fit_idx])
            elif name == "nwp_direct":
                if "fut_nwp_csi" not in W:
                    warnings.warn("nwp_direct: no fut_nwp_csi in windows, "
                                  "skipping")
                    continue
                b.fit(W["fut_csi"][fit_idx], W["fut_mask"][fit_idx])
            else:
                b.fit(W["fut_csi"][fit_idx], W["fut_mask"][fit_idx])
            fitted[name] = b
        except Exception as e:                          # pragma: no cover
            warnings.warn(f"baseline '{name}' failed to fit: {e}")
    return fitted


# ============================================================================
# ===== SECTION: capability guards -- forbid what a checkpoint can't do =====
# ============================================================================
# Availability variants used to only WARN after computing a nonsense forecast
# (see the OOD note in _warn_if_ood below). That's backwards: if a checkpoint
# was never trained to cope with a condition, don't produce and plot a
# forecast for it at all by default. Every capability check here is read off
# the checkpoint's OWN training config -- never guessed from its output --
# and every guard can be bypassed with one explicit flag when you genuinely
# want the diagnostic (e.g. to see HOW it fails), at which point the output
# is loudly labeled as unsupported/diagnostic rather than presented as a
# forecast.
class UnsupportedByCheckpoint(RuntimeError):
    """Raised when a requested operation asks a checkpoint to do something
    its own training config shows it was never prepared for. Pass
    --allow-unsupported to run it anyway -- the result is a diagnostic probe
    of off-manifold behavior, not a sound forecast."""


def capability_report(fm) -> Dict[str, bool]:
    """What this checkpoint can and cannot be asked to do, read directly off
    its frozen training config (fm.cfg, fm.nwp_spec, fm.site_vec, fm.prior)
    -- the same information the checkpoint carries when loaded, so this is
    exact, not a guess about behavior.

    NWP enters this codebase two SEPARATE ways (fullCode_v2.py's own module
    docstring, section on NWP): (A) as extra per-position conditioning
    channels (model.condition_nwp), and (B) as the source of the 'nwp'
    PRIOR's anchor (DayPrior.needs_nwp, i.e. prior.kind == 'nwp'). These have
    very different absence-tolerance, so they get separate fields instead of
    being collapsed into one 'has_nwp':

    - Conditioning channels (A) are architecturally absence-safe BY DESIGN,
      not just by luck: FlowMatcher._fut_extras() zero-fills any NWP channel
      that isn't supplied, including the 'present' flag going to 0 -- the
      exact signal the model already sees whenever real HRRR data has a gap
      at inference *and* training time (fullCode_v2.py: 'Both doors are
      optional and degrade gracefully to the no-NWP behaviour when the
      columns are absent.'). So has_nwp_conditioning alone does not make
      no_nwp unsupported.
    - The 'nwp' prior anchor (B) is where a checkpoint can genuinely be
      "anchored to NWP": DayPrior._anchor() falls back to the clear-sky
      anchor when fut_nwp_anchor is None rather than crashing, but that is a
      real change of source distribution, not a no-op. Only checkpoints
      trained with prior.kind == 'nwp' (DayPrior.needs_nwp) carry this
      dependency -- every other prior (clearsky/climatology/persistence/
      blend/white) never reads fut_nwp at all in _anchor().
    """
    t = fm.cfg.get("train", {})
    prior = getattr(fm, "prior", None)     # None for DeepQuantile checkpoints
    return {
        # train.history_dropout>0 means SOME batches trained with history
        # fully blanked -- the only regime this codebase regularizes for.
        "history_dropout_trained": float(t.get("history_dropout", 0.0) or 0.0) > 0.0,
        "has_nwp": bool(getattr(fm, "nwp_spec", [])),
        # kept: True iff the checkpoint's prior anchor is 'nwp' -- read via
        # DayPrior.needs_nwp, not guessed. This, not has_nwp, is what "the
        # checkpoint depends on NWP" actually means for whether it can run
        # without it.
        "nwp_anchored_prior": bool(getattr(prior, "needs_nwp", False)),
        # there is no NWP-dropout knob anywhere in this codebase (grep
        # train.* for it) -- so this is always False today. Kept as an
        # explicit field, not a bare constant, so that IF an nwp_dropout
        # regularizer is ever added to fit(), this flips on automatically
        # and the guards below start allowing it without any edits here.
        "nwp_dropout_trained": float(t.get("nwp_dropout", 0.0) or 0.0) > 0.0,
        "site_conditioned": fm.site_vec is not None,
        "multi_site": bool(fm.cfg.get("sites")),
    }


def _nwp_absence_ok(cap: Dict[str, bool]) -> bool:
    """True if it's safe to compute a variant with NWP fully absent for this
    checkpoint by default (no diagnostic label needed): either it was
    explicitly trained for it (nwp_dropout_trained -- doesn't exist in this
    codebase today, kept for forward-compat), or its prior isn't NWP-
    anchored, in which case NWP is only a conditioning channel that
    architecturally zero-fills (see capability_report's docstring)."""
    return cap["nwp_dropout_trained"] or not cap["nwp_anchored_prior"]


def _require(ok: bool, message: str, allow_unsupported: bool):
    if ok:
        return
    if allow_unsupported:
        warnings.warn(
            f"PROCEEDING PAST AN UNSUPPORTED OPERATION ({message}) because "
            f"--allow-unsupported was set. Treat this output as a "
            f"diagnostic probe of off-manifold behavior, not a forecast.",
            RuntimeWarning)
        return
    raise UnsupportedByCheckpoint(
        f"{message} Pass --allow-unsupported to run it anyway (diagnostic "
        f"only -- see capability_report()/the availability_variants "
        f"docstring for why this is blocked by default).")


# ============================================================================
# ===== SECTION: availability variants =======================================
# ============================================================================
def zero_history(hist_csi, hist_zen, hist_mask):
    """Blank history the way train.history_dropout blanks it: content AND
    the zenith channel AND the mask, all zeroed/all-False. Matches
    FlowMatcher.fit's dropout exactly (hist_t *= 0; hm_t &= False) so
    'no_history' at inference reproduces a condition the net actually trained
    under, rather than a new, unseen partial-blank state."""
    hist_csi = np.zeros_like(np.asarray(hist_csi, np.float32))
    hist_zen = np.zeros_like(np.asarray(hist_zen, np.float32))
    hist_mask = np.zeros_like(np.asarray(hist_mask, bool))
    return hist_csi, hist_zen, hist_mask


def availability_variants(fm, W: Dict[str, np.ndarray], rows: np.ndarray,
                          n_ensemble=100, seed=0, nwp_fill="zero",
                          allow_unsupported=False):
    """Forecast `rows` under every information condition this checkpoint can
    ACTUALLY be asked about, per capability_report(). Variants the checkpoint
    was never trained for are SKIPPED by default (reported, not silently
    dropped) rather than computed and plotted as if they were forecasts --
    pass allow_unsupported=True to compute them anyway as labeled
    diagnostics.

    Guard rules (see capability_report):
      'no_history'         needs history_dropout_trained
      'no_nwp'/'..._no_nwp' need nwp_dropout_trained (today: no checkpoint in
                            this codebase has this, since there is no such
                            training knob -- so these are blocked unless you
                            explicitly override, every time, until fit() grows
                            an nwp_dropout regularizer)

    Returns (variants, skipped): variants is {name: ensemble}; skipped is
    {name: reason} for anything not computed.
    """
    hist_csi, hist_zen, hist_mask = (W["hist_csi"][rows], W["hist_zen"][rows],
                                     W["hist_mask"][rows])
    fut_zen, fut_mask = W["fut_zen"][rows], W["fut_mask"][rows]
    gcs = W["fut_ghi_cs"][rows] if "fut_ghi_cs" in W else None
    site_coords = W["site_coords"][rows] if "site_coords" in W else None
    fut_nwp = slice_nwp_from_windows(W, rows)  # defined later in this module;
                                               # resolved at call time, not here

    cap = capability_report(fm)
    has_nwp = cap["has_nwp"] and fut_nwp is not None
    z_csi, z_zen, z_mask = zero_history(hist_csi, hist_zen, hist_mask)

    candidates = {
        "full": (dict(hist_csi=hist_csi, hist_zen=hist_zen,
                     hist_mask=hist_mask, fut_nwp=fut_nwp), True, ""),
        "no_history": (dict(hist_csi=z_csi, hist_zen=z_zen, hist_mask=z_mask,
                            fut_nwp=fut_nwp), cap["history_dropout_trained"],
                      "checkpoint's train.history_dropout was 0/absent, so "
                      "it never trained without history"),
    }
    if has_nwp:
        nwp_ok = _nwp_absence_ok(cap)
        nwp_reason = (
            "checkpoint's prior is NWP-anchored (prior.kind=='nwp'); "
            "removing NWP falls back to a clear-sky anchor rather than "
            "crashing, but that's a real change of source distribution, "
            "and no NWP-dropout training exists to cover it, so it's "
            "blocked by default" if cap["nwp_anchored_prior"] else
            "checkpoint has no NWP-dropout training, but its prior isn't "
            "NWP-anchored, so removing NWP only zero-fills the "
            "conditioning channels -- architecturally safe by design "
            "(see capability_report docstring)")
        candidates["no_nwp"] = (
            dict(hist_csi=hist_csi, hist_zen=hist_zen, hist_mask=hist_mask,
                fut_nwp=None), nwp_ok, nwp_reason)
        both_ok = cap["history_dropout_trained"] and nwp_ok
        candidates["no_history_no_nwp"] = (
            dict(hist_csi=z_csi, hist_zen=z_zen, hist_mask=z_mask,
                fut_nwp=None), both_ok,
            "needs history_dropout training AND (nwp_dropout training OR a "
            "non-NWP-anchored prior); checkpoint has " +
            ("history_dropout but an NWP-anchored prior with no "
             "NWP-dropout training" if cap["history_dropout_trained"]
             and not nwp_ok else
             "a non-NWP-anchored prior but no history_dropout training"
             if nwp_ok else "neither"))
        if nwp_fill == "neutral" and gcs is not None:
            # diagnostic-only by construction (present=1 is a false claim),
            # so it never claims to be supported -- always requires the flag
            candidates["no_nwp_neutral"] = (
                dict(hist_csi=hist_csi, hist_zen=hist_zen,
                    hist_mask=hist_mask,
                    fut_nwp=neutral_nwp_fill(fm, gcs)), False,
                "diagnostic-only variant (claims NWP present=1 with fake "
                "neutral content) -- always requires --allow-unsupported")

    out, skipped = {}, {}
    for name, (kw, supported, reason) in candidates.items():
        if not supported:
            if not allow_unsupported:
                skipped[name] = reason
                continue
            warnings.warn(
                f"variant '{name}': PROCEEDING PAST UNSUPPORTED ({reason}) "
                f"because --allow-unsupported was set; treat as a "
                f"diagnostic probe, not a forecast.", RuntimeWarning)
        rng = np.random.default_rng(seed)
        ens = fm.predict_ensemble(
            kw["hist_csi"], kw["hist_zen"], fut_zen, kw["hist_mask"], fut_mask,
            fut_ghi_cs=gcs, fut_nwp=kw["fut_nwp"], n_ensemble=n_ensemble,
            rng=rng, site_coords=site_coords)
        _warn_if_ood(name, ens, fut_mask, gcs, fm.cfg)
        out[name] = ens
    return out, skipped


def neutral_nwp_fill(fm, fut_ghi_cs):
    """Build an NWP dict that claims presence (present=1) but supplies
    climatologically neutral content instead of zero: dswrf ~ clear-sky GHI
    (implies CSI=1, i.e. "assume typical/clear"), cloud cover at 50%, NWP-CSI
    at 1.0. Purely diagnostic (real absence should carry present=0) -- see
    availability_variants' docstring for why this variant exists."""
    N, H = np.asarray(fut_ghi_cs).shape
    out = {}
    for key, kind in getattr(fm, "nwp_spec", []):
        if kind == "present":
            out[key] = np.ones((N, H), np.float32)
        elif kind == "ghi_norm":
            out[key] = np.asarray(fut_ghi_cs, np.float32)
        elif kind == "unit":
            out[key] = np.full((N, H), 50.0, np.float32)   # 50% cloud cover
        else:                                              # "csi"
            out[key] = np.ones((N, H), np.float32)          # CSI = 1 (clear)
    return out


def _warn_if_ood(name, ens, fut_mask, fut_ghi_cs, cfg, csi_alarm=1.25):
    """Flag predictions that land far outside a physically ordinary CSI range
    on a meaningful fraction of valid steps -- the signature of a model
    extrapolating outside its training distribution (see
    availability_variants' docstring). Doesn't raise or alter the forecast,
    just makes an easy-to-miss problem loud."""
    if name == "full":
        return
    mask = np.asarray(fut_mask, bool)
    if not mask.any():
        return
    med = np.median(ens, axis=1)                          # [N,H] physical CSI
    bad = (med > csi_alarm) & mask
    frac = bad.sum() / mask.sum()
    if frac > 0.15:
        csi_max = cfg.get("data", {}).get("csi_max", 1.8)
        warnings.warn(
            f"variant '{name}': median CSI exceeds {csi_alarm} on "
            f"{frac:.0%} of valid steps (peak {med[mask].max():.2f}, "
            f"training csi_max={csi_max}). This checkpoint has no dropout "
            f"regularizer for whatever input you removed (see "
            f"availability_variants docstring) -- treat this variant as an "
            f"out-of-distribution extrapolation, not a reliable forecast.",
            RuntimeWarning)


# ============================================================================
# ===== SECTION: cold-start geometry from coordinates + clear-sky model =====
# ============================================================================
def geometry_from_coords(lat: float, lon: float, alt: float,
                         start_date: str, K: int, n_days: int,
                         resolution_min: float = 10.0,
                         noon_col: Optional[int] = None):
    """Compute fut_zen / fut_mask / fut_ghi_cs for `n_days` calendar days
    starting at `start_date`, purely from astronomy (pvlib solar position +
    the Ineichen clear-sky model with climatological Linke turbidity) --
    exactly what is knowable about a location before any sensor or NWP data
    exists. Output is noon-centered on `noon_col` (default K//2, since a
    cold-start forecast has no historical grid to align to -- any consistent
    noon-centering works as long as it's used consistently, which it is
    here). Steps outside +/-zenith_daylight_max degrees are masked False,
    matching how preprocess() builds the daylight mask.

    Returns dict with fut_zen, fut_mask, fut_ghi_cs, each [1, K*n_days], plus
    the site_coords row [1,3] normalized the same way `_site_coord_vec` does
    (only meaningful for checkpoints with condition_site on).
    """
    import pvlib
    noon_col = K // 2 if noon_col is None else int(noon_col)
    zen_max = 85.0

    steps_per_day = int(round(1440.0 / resolution_min))
    if steps_per_day < K:
        raise ValueError(
            f"resolution_min={resolution_min} gives only {steps_per_day} "
            f"steps/day, fewer than K={K}; use a finer resolution.")

    start = dt.date.fromisoformat(start_date)
    fut_zen = np.full(K * n_days, np.nan, np.float32)
    fut_ghi_cs = np.full(K * n_days, np.nan, np.float32)
    fut_mask = np.zeros(K * n_days, bool)

    tz_hours = round(lon / 15.0)                          # local-solar-ish tz
    for d in range(n_days):
        day = start + dt.timedelta(days=d)
        # build a UTC timestamp grid, shifted so local solar noon lands near
        # the middle of the day (mirrors the training-side noon centering)
        times = pd_date_range_utc(day, steps_per_day, tz_hours)
        loc = pvlib.location.Location(lat, lon, altitude=alt)
        solpos = loc.get_solarposition(times)
        zen = solpos["apparent_zenith"].to_numpy()
        cs = loc.get_clearsky(times, model="ineichen")
        ghi_cs = cs["ghi"].to_numpy()

        noon_step = int(np.argmin(zen))
        # place this day's steps on the shared K-wide, noon-centered row
        for s in range(steps_per_day):
            col = noon_col + (s - noon_step)
            if 0 <= col < K:
                sl = d * K + col
                fut_zen[sl] = zen[s]
                fut_ghi_cs[sl] = max(ghi_cs[s], 0.0)
                fut_mask[sl] = zen[s] < zen_max
    return {
        "fut_zen": fut_zen[None, :], "fut_mask": fut_mask[None, :],
        "fut_ghi_cs": fut_ghi_cs[None, :],
        "site_coords": None,   # caller fills via core._site_coord_vec if needed
    }


def pd_date_range_utc(day: dt.date, steps_per_day: int, tz_hours: int):
    import pandas as pd
    minutes = 1440 // steps_per_day
    start_local = dt.datetime(day.year, day.month, day.day)
    start_utc = start_local - dt.timedelta(hours=tz_hours)
    return pd.date_range(start_utc, periods=steps_per_day,
                         freq=f"{minutes}min", tz="UTC")


def coldstart_forecast(fm, lat: float, lon: float, alt: float,
                       start_date: str, n_ensemble=100, seed=0,
                       resolution_min=10.0, noon_col=None,
                       allow_unsupported=False):
    """Forecast for a bare location with NO history and NO NWP -- geometry +
    clear-sky only. Returns (ensemble [1,M,H_out], geometry dict) so the
    caller can plot it with plot_forecast(mode='ghi', ghi_cs_row=...).

    Guarded by capability_report(): a cold start is inherently a
    'no_history' + 'no_nwp' request (there IS no history or NWP for a
    location before any data exists), so it is only allowed by default when
    the checkpoint was trained with history_dropout AND NWP absence is safe
    for it -- either no NWP conditioning at all, NWP-dropout training (does
    not exist in this codebase today), or a prior that isn't NWP-anchored
    (see capability_report()/_nwp_absence_ok() docstrings: only prior.kind
    == 'nwp' actually depends on NWP being present; every other prior
    ignores fut_nwp entirely, and the conditioning channels zero-fill by
    design regardless of prior). Pass allow_unsupported=True to run it
    anyway as a labeled diagnostic.
    """
    cap = capability_report(fm)
    _require(cap["history_dropout_trained"],
            "coldstart has zero history, but this checkpoint's "
            "train.history_dropout was 0/absent", allow_unsupported)
    _require(not cap["has_nwp"] or _nwp_absence_ok(cap),
            "coldstart has no NWP, but this checkpoint's prior is "
            "NWP-anchored (prior.kind=='nwp') and no NWP-dropout "
            "regularizer exists to cover full NWP absence", allow_unsupported)

    K, n_days = fm.K, fm.n_days
    geo = geometry_from_coords(lat, lon, alt, start_date, K, n_days,
                               resolution_min=resolution_min,
                               noon_col=noon_col)
    H_in = fm.H_in
    hist_csi = np.zeros((1, H_in), np.float32)
    hist_zen = np.zeros((1, H_in), np.float32)
    hist_mask = np.zeros((1, H_in), bool)

    site_coords = None
    if fm.site_vec is not None:
        if cap["multi_site"]:
            # trained on varying per-window coordinates -- a new (lat,lon)
            # is exactly the kind of input it learned to consume
            site_coords = core._site_coord_vec(fm.cfg, lat, lon, alt)[None, :]
        else:
            # single-site: site_vec is a CONSTANT frozen at training time.
            # Feeding a different coordinate here wouldn't be read as "a new
            # site" -- it would be an unseen value on a channel the net only
            # ever saw as one fixed constant. Use the frozen vector (still
            # correct for the trained site) and say so; geometry (zenith,
            # clear sky) is still computed for the coordinates you gave,
            # since that part is pure astronomy and always valid anywhere.
            warnings.warn(
                "this checkpoint is single-site-conditioned (site_vec is a "
                "constant frozen at training); your --lat/--lon are used "
                "for the SOLAR GEOMETRY only, the site-conditioning channel "
                "still carries the training site's coordinates. Retrain in "
                "multi-site mode (cfg.sites) for the model to actually "
                "generalize to new locations.", RuntimeWarning)

    rng = np.random.default_rng(seed)
    ens = fm.predict_ensemble(hist_csi, hist_zen, geo["fut_zen"], hist_mask,
                              geo["fut_mask"], fut_ghi_cs=geo["fut_ghi_cs"],
                              fut_nwp=None, n_ensemble=n_ensemble, rng=rng,
                              site_coords=site_coords)
    _warn_if_ood("coldstart", ens, geo["fut_mask"], geo["fut_ghi_cs"], fm.cfg)
    return ens, geo



# ============================================================================
# ===== SECTION: autoregressive vs direct rollout ============================
# ============================================================================
def _find_block_rows(W: Dict[str, np.ndarray], start_row: int, n_blocks: int,
                     n_days: int) -> List[int]:
    """Consecutive, NON-overlapping n_days-day blocks starting at start_row,
    found by matching first_day_ord (windows are daily-stride, so a block's
    forecast horizon is exactly the next block's most-recent history)."""
    fdo = W["first_day_ord"]
    rows = [int(start_row)]
    target = int(fdo[start_row]) + n_days
    for _ in range(n_blocks - 1):
        hit = np.where(fdo == target)[0]
        if len(hit) == 0:
            raise ValueError(
                f"no window starts at day_ord {target} (need daily-stride, "
                f"non-gapped windows for a clean autoregressive chain); "
                f"rollout stops after {len(rows)} block(s)")
        rows.append(int(hit[0]))
        target += n_days
    return rows


def rollout_forecast(fm, W: Dict[str, np.ndarray], start_row: int,
                     n_blocks: int, mode: str = "autoregressive",
                     n_ensemble=100, seed=0, feedback="median"):
    """Chain forecasts across `n_blocks` consecutive forecast-day blocks.

    mode='direct'         : each block forecast from its own TRUE history
                             (teacher forcing).
    mode='autoregressive'  : block 0 uses true history; every later block's
                             history has its most recent n_days*K steps
                             REPLACED by the previous block's own forecast.
    feedback='median'     : feed forward the ensemble median (fast, one
                             trajectory).
    feedback='ensemble'   : feed forward all M members independently (each
                             ensemble path is rolled forward on its own,
                             giving M genuinely different future histories --
                             slower but statistically honest compounding).

    Returns dict: {"rows": [...], "pred": [n_blocks, M, H_out] or
    [n_blocks, M, H_out] per-path for ensemble feedback, "truth":
    [n_blocks, H_out], "mask": [n_blocks, H_out]}.
    """
    if mode not in ("direct", "autoregressive"):
        raise ValueError("mode must be 'direct' or 'autoregressive'")
    K, n_days = fm.K, fm.n_days
    rows = _find_block_rows(W, start_row, n_blocks, n_days)
    gcs = W["fut_ghi_cs"] if "fut_ghi_cs" in W else None

    preds, truths, masks = [], [], []
    hist_csi = W["hist_csi"][rows[0]].copy()
    hist_zen = W["hist_zen"][rows[0]].copy()
    hist_mask = W["hist_mask"][rows[0]].copy()
    M = int(n_ensemble)
    # ensemble-feedback keeps M independent trajectories; point-feedback keeps 1
    n_paths = M if (mode == "autoregressive" and feedback == "ensemble") else 1
    hist_csi = np.tile(hist_csi[None], (n_paths, 1))
    hist_zen = np.tile(hist_zen[None], (n_paths, 1))
    hist_mask = np.tile(hist_mask[None], (n_paths, 1))

    for bi, r in enumerate(rows):
        fut_nwp = slice_nwp_from_windows(W, np.array([r]))
        if fut_nwp is not None:
            fut_nwp = {k: np.tile(v, (n_paths, 1)) for k, v in fut_nwp.items()}
        fut_zen = np.tile(W["fut_zen"][r][None], (n_paths, 1))
        fut_mask = np.tile(W["fut_mask"][r][None], (n_paths, 1))
        fgcs = (np.tile(gcs[r][None], (n_paths, 1)) if gcs is not None else None)

        if mode == "direct":
            # ignore the chained history: always the window's own true one
            h_csi = np.tile(W["hist_csi"][r][None], (n_paths, 1))
            h_zen = np.tile(W["hist_zen"][r][None], (n_paths, 1))
            h_mask = np.tile(W["hist_mask"][r][None], (n_paths, 1))
        else:
            h_csi, h_zen, h_mask = hist_csi, hist_zen, hist_mask

        rng = np.random.default_rng(seed + bi)
        block_ens = np.stack([
            fm.predict_ensemble(h_csi[[p]], h_zen[[p]], fut_zen[[p]],
                                h_mask[[p]], fut_mask[[p]],
                                fut_ghi_cs=(None if fgcs is None else fgcs[[p]]),
                                fut_nwp=(None if fut_nwp is None else
                                        {k: v[[p]] for k, v in fut_nwp.items()}),
                                n_ensemble=(1 if n_paths > 1 else M),
                                rng=rng)[0]
            for p in range(n_paths)], axis=0)               # [n_paths, m, H]
        preds.append(block_ens.reshape(-1, block_ens.shape[-1])
                    if n_paths == 1 else block_ens)

        if "fut_csi" in W:
            truths.append(W["fut_csi"][r])
            masks.append(W["fut_mask"][r])

        if mode == "autoregressive" and bi < len(rows) - 1:
            H_in = h_csi.shape[1]
            keep = H_in - n_days * K
            if n_paths == 1:
                fb = (np.median(block_ens[0], axis=0) if feedback == "median"
                     else block_ens[0][0])
                fb = np.tile(fb[None], (n_paths, 1))
            else:
                fb = block_ens[:, 0, :]                     # [n_paths, H]
            new_hist_csi = np.concatenate([hist_csi[:, n_days * K:], fb],
                                          axis=1)
            new_hist_zen = np.concatenate(
                [hist_zen[:, n_days * K:],
                 np.tile(W["fut_zen"][r][None], (n_paths, 1))], axis=1)
            # Preserve the true daylight/padding mask when feeding the previous
            # forecast back as history. Padded slots must not become observations.
            feedback_mask = np.tile(W["fut_mask"][r][None, :], (n_paths, 1))
            new_hist_mask = np.concatenate(
                [hist_mask[:, n_days * K:], feedback_mask], axis=1)
            hist_csi, hist_zen, hist_mask = (new_hist_csi, new_hist_zen,
                                             new_hist_mask)

    return {
        "rows": rows,
        "pred": np.stack(preds, axis=0),                    # [n_blocks, M, H]
        "truth": (np.stack(truths, axis=0) if truths else None),
        "mask": (np.stack(masks, axis=0) if masks else None),
        "K": K, "n_days": n_days,
    }


def rollout_forecast_days(fm, W: Dict[str, np.ndarray], start_row: int,
                          n_days_requested: int,
                          mode: str = "autoregressive", n_ensemble=100,
                          seed=0, feedback="median"):
    """Forecast exactly ``n_days_requested`` days by chaining model blocks.

    A trained checkpoint predicts ``fm.n_days`` days per call. This helper
    converts the user-facing request into the required number of complete
    blocks, runs the existing day-by-day autoregressive rollout, concatenates
    the generated blocks, and trims the returned trajectory to exactly the
    requested horizon. For ``mode='autoregressive'``, every block after the
    first receives the previous block's generated forecast as history; no
    later block is silently re-grounded in observations.

    The returned dictionary contains ``pred`` with shape ``[M, K*n_days]``,
    plus concatenated truth/mask arrays and the source block rows. A request
    that cannot be supported by the saved daily windows raises a clear error
    before expensive inference begins.
    """
    requested = int(n_days_requested)
    if requested < 1:
        raise ValueError("n_days_requested must be at least 1")
    block_days = int(fm.n_days)
    n_blocks = int(math.ceil(requested / block_days))
    raw = rollout_forecast(fm, W, int(start_row), n_blocks,
                           mode=mode, n_ensemble=int(n_ensemble),
                           seed=int(seed), feedback=feedback)
    trim = requested * int(fm.K)
    pred = np.concatenate([raw["pred"][i] for i in range(n_blocks)], axis=-1)
    pred = pred[..., :trim]
    truth = None
    if raw["truth"] is not None:
        truth = np.concatenate([raw["truth"][i] for i in range(n_blocks)], axis=-1)[:trim]
    mask = None
    if raw["mask"] is not None:
        mask = np.concatenate([raw["mask"][i] for i in range(n_blocks)], axis=-1)[:trim]
    return {
        "rows": raw["rows"], "pred": pred, "truth": truth, "mask": mask,
        "K": int(fm.K), "n_days": requested, "n_blocks": n_blocks,
    }


# ============================================================================
# ===== SECTION: windows / NWP helpers (kept compatible with the old file) ==
# ============================================================================
def slice_nwp_from_windows(W, rows):
    nwp_keys = [k for k in W if k.startswith("fut_")
                and k not in ("fut_csi", "fut_zen", "fut_mask", "fut_ghi_cs")]
    if not nwp_keys:
        return None
    return {k: W[k][rows] for k in nwp_keys}


def load_windows(path):
    return dict(np.load(path))


def forecast_from_arrays(fm, hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                         n_ensemble=100, seed=0, fut_ghi_cs=None, fut_nwp=None,
                         site_coords=None):
    spec = getattr(fm, "nwp_spec", [])
    if spec and not fut_nwp:
        warnings.warn(
            f"checkpoint was trained WITH NWP conditioning ({len(spec)} "
            f"channels) but no fut_nwp was passed; NWP slots zero-filled, "
            f"forecast will be DEGRADED.", RuntimeWarning)
    rng = np.random.default_rng(seed)
    return fm.predict_ensemble(hist_csi, hist_zen, fut_zen, hist_mask, fut_mask,
                               fut_ghi_cs=fut_ghi_cs, fut_nwp=fut_nwp,
                               n_ensemble=n_ensemble, rng=rng,
                               site_coords=site_coords)


# ============================================================================
# ===== SECTION: plotting =====================================================
# ============================================================================
def _extend_clearsky_envelope(gcs_row, mask_row, K, n_days, zenith_row=None,
                               zenith_cutoff=85.0):
    """Complete clear-sky reference values only in valid daylight geometry.

    At solar zenith > 85 degrees, no polynomial clear-sky completion is
    allowed: the returned reference is NaN and the model output remains the
    only plotted forecast contribution at that position.
    """
    H = K * n_days
    out = np.full(H, np.nan)
    for b in range(n_days):
        sl = slice(b * K, (b + 1) * K)
        g = np.asarray(gcs_row[sl], float)
        m = np.asarray(mask_row[sl], bool) & np.isfinite(g)
        if zenith_row is not None:
            z = np.asarray(zenith_row[sl], float)
            daylight = np.isfinite(z) & (z <= float(zenith_cutoff))
            m &= daylight
        if m.sum() < 3:
            out[sl] = np.where(m, g, np.nan)
            continue
        x = np.arange(K)
        xv = x[m]; gv = g[m]
        try:
            coef = np.polyfit(xv, gv, 2)
            fit = np.clip(np.polyval(coef, x), 0.0, None)
            filled = np.where(m, g, fit)
        except np.linalg.LinAlgError:
            filled = np.where(m, g, np.nan)
        if zenith_row is not None:
            filled[~daylight] = np.nan
        out[sl] = filled
    return out


def zenith_cutoff_from_config(config_or_model, default=85.0):
    """Read the daylight zenith cutoff from the trained data configuration."""
    cfg = getattr(config_or_model, "cfg", config_or_model) or {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    try:
        return float(data_cfg.get("zenith_daylight_max", default))
    except (TypeError, ValueError):
        return float(default)


# ----------------------------------------------------------------------------
# Theme -- shared with the Streamlit app's CSS palette, so a chart looks like
# it belongs on the page instead of a plot dropped on top of it. Charts are
# rendered with a transparent background (no axes box, no figure facecolor)
# so whatever card/background the app places behind them shows through.
# ----------------------------------------------------------------------------
THEME = {
    "bg1": "#0c1929",
    "grid": "#1a2e48",
    "grid2": "#243d5c",
    "text": "#f0f4f8",
    "muted": "#9db4c8",
    "dim": "#5a7a96",
    "amber": "#e8b84b",
    "amber2": "#c9963a",
    "blue": "#5b9bd5",
    "green": "#7fbf7f",
    "legend_bg": "#0c1929",
}
# primary series (the one with full shading) is always amber; comparison
# series cycle through the rest, chosen to stay readable on a dark background
_PALETTE = [THEME["amber"], THEME["blue"], "#7fbf7f", "#e0798a",
           "#b48ead", "#6cc0c4", "#e0c46c", "#c98f5e"]


def _apply_theme(fig, ax):
    """Transparent figure/axes, light text and grid tuned to the app's dark
    theme. Called once per chart right before drawing."""
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    for spine in ax.spines.values():
        spine.set_color(THEME["grid2"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=THEME["muted"], labelsize=9)
    ax.xaxis.label.set_color(THEME["muted"])
    ax.yaxis.label.set_color(THEME["muted"])
    ax.title.set_color(THEME["text"])
    ax.grid(True, color=THEME["grid"], linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)


def plot_comparison(series: Dict[str, Optional[np.ndarray]], truth_row,
                    mask_row, K, n_days, title, out_path,
                    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), mode="csi",
                    ghi_cs_row=None, zenith_row=None, zenith_cutoff=85.0,
                    primary: Optional[str] = None,
                    spaghetti=False, spaghetti_alpha=0.05, spaghetti_max=60,
                    style: Optional[str] = None):
    """Fan chart for a PRIMARY series (full quantile shading) plus any number
    of comparison series (median-only line each, distinct color).

    series : {label: pred_ens [M,H_out] or None}. `primary` selects which key
             gets the shaded fan; defaults to the first key. A comparison
             series with only 1 member (a point-forecast baseline like
             day_persistence or nwp_direct) still works -- its "median" is
             just that one value.
    style : high-level display choice for the PRIMARY series --
             "bands" (default): shaded quantile bands only.
             "scenarios": individual simulated trajectories only, no bands.
             None: legacy behaviour, controlled by `spaghetti` below.
    spaghetti : legacy per-flag control, used only when `style` is None: if
             True, overlay up to `spaghetti_max` individual ensemble member
             trajectories on top of the shaded bands. Kept for callers (the
             CLI, the self-test) that don't use `style`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    labels = list(series.keys())
    if not labels:
        raise ValueError("plot_comparison needs at least one series")

    # Resolve renamed UI labels defensively. The app uses an exact scientific
    # label such as `FlowMatcher · rep=...`, while older callers may still pass
    # `Main model`, `flow`, `full`, or `forecast`.
    requested_primary = primary
    if primary not in series or series.get(primary) is None:
        aliases = {
            "Main model": ("flow", "full", "forecast", "direct"),
            "FlowMatcher": ("flow", "full", "forecast"),
            "flow": ("full", "forecast"),
            "full": ("forecast",),
        }
        candidates = list(aliases.get(primary, ()))
        candidates += [k for k in labels if any(token in str(k) for token in ("FlowMatcher", "flow", "Main model"))]
        candidates += labels
        resolved = next((k for k in candidates if k in series and series.get(k) is not None), None)
        if resolved is None:
            raise ValueError(f"primary series '{requested_primary}' missing or None; available series: {labels}")
        primary = resolved

    draw_bands = True
    draw_spaghetti = bool(spaghetti)
    if style == "scenarios":
        draw_bands = False
        draw_spaghetti = True
    elif style == "bands":
        draw_bands = True
        draw_spaghetti = False

    H = series[primary].shape[1]
    x = np.arange(H)
    valid = np.asarray(mask_row, bool)

    def m(a):
        a = np.asarray(a, float).copy(); a[~valid] = np.nan; return a

    def to_plot(v):
        if v is None:
            return None
        if mode == "ghi":
            if ghi_cs_row is None:
                raise ValueError("mode='ghi' requires ghi_cs_row")
            return v * np.asarray(ghi_cs_row, float)[None, :]
        return v

    def plot_spaghetti(ax, v_plot, color, label_prefix=""):
        """Draw up to spaghetti_max member trajectories from v_plot [M,H],
        evenly sampled across the ensemble (not just the first N) so the
        overlay reflects the full spread, not an arbitrary slice."""
        M = v_plot.shape[0]
        if M <= 1:
            return
        n = min(spaghetti_max, M)
        idx = np.linspace(0, M - 1, n).round().astype(int)
        for j in idx:
            ax.plot(x, m(v_plot[j]), color=color, lw=1.05,
                   alpha=spaghetti_alpha, zorder=1.5)
        # one representative proxy line for the legend, not one per member
        ax.plot([], [], color=color, lw=2.0,
               alpha=min(spaghetti_alpha * 6, 0.75),
               label=f"{label_prefix}individual scenarios")

    gcs = np.asarray(ghi_cs_row, float) if (mode == "ghi" and
                                            ghi_cs_row is not None) else None
    env = (_extend_clearsky_envelope(
                gcs, valid, K, n_days, zenith_row=zenith_row,
                zenith_cutoff=zenith_cutoff)
          if (mode == "ghi" and gcs is not None) else None)
    ylabel = "GHI (W/m$^2$)" if mode == "ghi" else "Clear-sky index (CSI)"

    qs = sorted(quantiles)
    lo_qs = [q for q in qs if q < 0.5]
    hi_qs = [q for q in qs if q > 0.5][::-1]
    if len(lo_qs) != len(hi_qs):
        raise ValueError(f"quantiles must be symmetric around 0.5; got {qs}")

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    _apply_theme(fig, ax)

    # Legend swatches for nested bands must represent the color actually
    # visible in the plot. The 50% band is drawn on top of the 80% band, so
    # its displayed color is alpha-composited twice over the chart background.
    band_legend_handles = []
    band_rgb = np.asarray(to_rgba(THEME["legend_bg"])[:3], dtype=float)
    amber_rgb = np.asarray(to_rgba(THEME["amber"])[:3], dtype=float)

    prim_plot = to_plot(series[primary])
    qv = {q: np.quantile(prim_plot, q, axis=0) for q in qs}
    if draw_bands:
        for i, (ql, qh) in enumerate(zip(lo_qs, hi_qs)):
            alpha = 0.20 + 0.13 * i
            ax.fill_between(
                x, m(qv[ql]), m(qv[qh]), alpha=alpha,
                color=THEME["amber"], linewidth=0, zorder=1,
                label="_nolegend_",
            )
            # Composite in drawing order so the legend patch matches the
            # rendered band rather than showing an isolated transparent amber.
            band_rgb = alpha * amber_rgb + (1.0 - alpha) * band_rgb
            band_legend_handles.append(
                Patch(
                    facecolor=(*band_rgb, 1.0), edgecolor="none",
                    label=f"{primary}: {int(round((qh - ql) * 100))}% predictive interval",
                )
            )
    if draw_spaghetti:
        prefix = "" if len(labels) == 1 else f"{primary}: "
        plot_spaghetti(ax, prim_plot, THEME["amber"], prefix)
    med_key = 0.5 if 0.5 in qv else None
    prim_med = qv[med_key] if med_key is not None else prim_plot.mean(0)
    ax.plot(x, m(prim_med), color=THEME["amber"], lw=2.5, zorder=3,
           label=f"{primary} (typical outcome)")

    ci = 1
    for name in labels:
        if name == primary or series[name] is None:
            continue
        v = to_plot(series[name])
        color = _PALETTE[ci % len(_PALETTE)]
        if draw_spaghetti:
            plot_spaghetti(ax, v, color, f"{name}: ")
        med = np.median(v, axis=0) if v.shape[0] > 1 else v[0]
        ax.plot(x, m(med), color=color, lw=1.45, ls="--", zorder=3, label=name)
        ci += 1

    if truth_row is not None:
        t = to_plot(np.asarray(truth_row)[None, :])[0]
        ax.plot(x, m(t), color=THEME["text"], lw=1.7, zorder=4,
               label="what actually happened")

    if mode == "ghi" and env is not None:
        ax.plot(x, env, color=THEME["blue"], ls=":", lw=1.1,
               alpha=0.8, label="clear-sky reference")
    elif mode == "csi":
        ax.axhline(1.0, color=THEME["muted"], ls="--", lw=0.8, alpha=0.5,
                  label="perfectly clear sky")

    for b in range(1, n_days):
        ax.axvline(b * K - 0.5, color=THEME["grid2"], ls=":", lw=1.0)
        ax.text(b * K - 0.5, ax.get_ylim()[1], f" day {b+1}", va="top",
               ha="left", fontsize=8, color=THEME["dim"])
    ax.text(0, ax.get_ylim()[1], " day 1", va="top", ha="left", fontsize=8,
           color=THEME["dim"])

    # Apply the same axis language and geometry to CSI and GHI. Only the
    # physical y-axis label and the optional clear-sky reference differ.
    ax.set_xlabel("Forecast time step · days shown consecutively")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(bottom=0)
    # Include the complete horizon even when early/late values are masked or
    # NaN. Without this explicit bound, Matplotlib autoscaling starts at the
    # first finite curve value and makes the chart appear shifted to the right.
    ax.set_xlim(-0.5, H - 0.5)

    # Keep the legend completely outside the plotting rectangle. It adapts its
    # number of columns to the number of curves and grows the bottom margin so
    # long scientific labels never cover data or get clipped.
    handles, legend_labels = ax.get_legend_handles_labels()
    handles = band_legend_handles + handles
    legend_labels = [h.get_label() for h in band_legend_handles] + legend_labels
    n_items = len(legend_labels)
    # Use the full figure width before adding another legend row. This keeps
    # large model-comparison legends shallow instead of consuming half the
    # image height.
    ncol = min(max(n_items, 1), 4)
    nrows = int(np.ceil(n_items / max(ncol, 1)))
    bottom = min(0.38, 0.16 + 0.035 * nrows)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=bottom)
    leg = fig.legend(
        handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.018),
        ncol=ncol, fontsize=8.0, facecolor=THEME["legend_bg"],
        edgecolor=THEME["grid2"], framealpha=0.97, labelcolor=THEME["text"],
        handlelength=2.35, columnspacing=1.0, handletextpad=0.45,
        borderpad=0.55, labelspacing=0.5,
    )
    for text in leg.get_texts():
        text.set_color(THEME["text"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, transparent=True, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


# thin wrapper: single-series fan chart (kept for `forecast` subcommand parity
# with the original forecast_and_plot.py)
def plot_forecast(pred_row, truth_row, mask_row, K, n_days, title, out_path,
                  quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), mode="csi",
                  ghi_cs_row=None, nwp_csi_row=None, spaghetti=False,
                  spaghetti_alpha=0.05, spaghetti_max=60,
                  zenith_row=None, zenith_cutoff=85.0,
                  style: Optional[str] = None):
    series = {"forecast": pred_row}
    if nwp_csi_row is not None:
        series["weather forecast (NWP)"] = nwp_csi_row[None, :]
    return plot_comparison(series, truth_row, mask_row, K, n_days, title,
                           out_path, quantiles=quantiles, mode=mode,
                           ghi_cs_row=ghi_cs_row, zenith_row=zenith_row,
                           zenith_cutoff=zenith_cutoff, primary="forecast",
                           spaghetti=spaghetti, spaghetti_alpha=spaghetti_alpha,
                           spaghetti_max=spaghetti_max, style=style)


# ============================================================================
# ===== SECTION: CLI ==========================================================
# ============================================================================
def _load_model_and_windows(args):
    fm = load_any(args.checkpoint, device=core.get_device(args.device))
    W = load_windows(args.windows)
    if W["fut_mask"].shape[1] != fm.H_out or W["hist_mask"].shape[1] != fm.H_in:
        raise ValueError(
            f"windows H_in/H_out ({W['hist_mask'].shape[1]}/"
            f"{W['fut_mask'].shape[1]}) != model ({fm.H_in}/{fm.H_out})")
    return fm, W


def cmd_list(args):
    print_checkpoints(args.models_dir)


def cmd_forecast(args):
    fm, W = _load_model_and_windows(args)
    rows = np.asarray(args.rows, int)
    have_gcs = "fut_ghi_cs" in W
    if args.mode == "ghi" and not have_gcs:
        raise ValueError("mode='ghi' needs fut_ghi_cs in the windows file")
    gcs_rows = W["fut_ghi_cs"][rows] if have_gcs else None
    fut_nwp = slice_nwp_from_windows(W, rows)
    pred = forecast_from_arrays(fm, W["hist_csi"][rows], W["hist_zen"][rows],
                                W["fut_zen"][rows], W["hist_mask"][rows],
                                W["fut_mask"][rows], n_ensemble=args.n_ensemble,
                                seed=args.seed, fut_ghi_cs=gcs_rows,
                                fut_nwp=fut_nwp)
    truth = W.get("fut_csi")
    qs = fm.cfg["eval"].get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
    for i, r in enumerate(rows):
        t = truth[r] if truth is not None else None
        gcs = W["fut_ghi_cs"][r] if have_gcs else None
        out = os.path.join(args.out, f"forecast_row{r}_{args.mode}.png")
        plot_forecast(pred[i], t, W["fut_mask"][r], fm.K, fm.n_days,
                     title=f"row {r} forecast ({args.mode})", out_path=out,
                     quantiles=qs, mode=args.mode, ghi_cs_row=gcs,
                     spaghetti=args.spaghetti,
                     spaghetti_alpha=args.spaghetti_alpha,
                     spaghetti_max=args.spaghetti_max)
        print("  saved", out)


def cmd_availability(args):
    fm, W = _load_model_and_windows(args)
    rows = np.asarray(args.rows, int)
    have_gcs = "fut_ghi_cs" in W
    if args.mode == "ghi" and not have_gcs:
        raise ValueError("mode='ghi' needs fut_ghi_cs in the windows file")
    qs = fm.cfg["eval"].get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
    for r in rows:
        variants, skipped = availability_variants(
            fm, W, np.array([r]), n_ensemble=args.n_ensemble, seed=args.seed,
            nwp_fill=args.nwp_fill, allow_unsupported=args.allow_unsupported)
        for name, reason in skipped.items():
            print(f"  row {r}: SKIPPED '{name}' -- {reason} "
                  f"(pass --allow-unsupported to run it anyway)")
        series = {name: ens[0] for name, ens in variants.items()}
        t = W["fut_csi"][r] if "fut_csi" in W else None
        gcs = W["fut_ghi_cs"][r] if have_gcs else None
        out = os.path.join(args.out, f"availability_row{r}_{args.mode}.png")
        plot_comparison(series, t, W["fut_mask"][r], fm.K, fm.n_days,
                        title=f"row {r}: forecast under data availability",
                        out_path=out, quantiles=qs, mode=args.mode,
                        ghi_cs_row=gcs, primary="full",
                        spaghetti=args.spaghetti,
                        spaghetti_alpha=args.spaghetti_alpha,
                        spaghetti_max=args.spaghetti_max)
        print("  saved", out)


def cmd_coldstart(args):
    fm = load_any(args.checkpoint, device=core.get_device(args.device))
    ens, geo = coldstart_forecast(fm, args.lat, args.lon, args.alt,
                                  args.start_date, n_ensemble=args.n_ensemble,
                                  seed=args.seed,
                                  resolution_min=args.resolution_min,
                                  allow_unsupported=args.allow_unsupported)
    qs = fm.cfg["eval"].get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
    out = os.path.join(args.out, f"coldstart_{args.start_date}_{args.mode}.png")
    plot_forecast(ens[0], None, geo["fut_mask"][0], fm.K, fm.n_days,
                 title=f"cold start ({args.lat:.3f},{args.lon:.3f}) "
                       f"from {args.start_date}",
                 out_path=out, quantiles=qs, mode=args.mode,
                 ghi_cs_row=(geo["fut_ghi_cs"][0] if args.mode == "ghi"
                            else None),
                 spaghetti=args.spaghetti,
                 spaghetti_alpha=args.spaghetti_alpha,
                 spaghetti_max=args.spaghetti_max)
    print("  saved", out)


def cmd_rollout(args):
    fm, W = _load_model_and_windows(args)
    qs = fm.cfg["eval"].get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
    have_gcs = "fut_ghi_cs" in W
    modes = ["direct", "autoregressive"] if args.mode == "both" else [args.mode]
    results = {}
    for md in modes:
        results[md] = rollout_forecast(
            fm, W, args.start_row, args.n_blocks, mode=md,
            n_ensemble=args.n_ensemble, seed=args.seed,
            feedback=("ensemble" if args.rollout_ensemble else "median"))

    K, n_days = fm.K, fm.n_days
    n_blocks = args.n_blocks
    for bi in range(n_blocks):
        series = {}
        for md in modes:
            series[md] = results[md]["pred"][bi]
        r = results[modes[0]]["rows"][bi]
        t = results[modes[0]]["truth"][bi] if results[modes[0]]["truth"] is \
            not None else None
        mrow = results[modes[0]]["mask"][bi] if results[modes[0]]["mask"] is \
            not None else np.ones(K * n_days, bool)
        gcs = W["fut_ghi_cs"][r] if have_gcs else None
        out = os.path.join(args.out, f"rollout_block{bi}_row{r}_{args.plot_mode}.png")
        plot_comparison(series, t, mrow, K, n_days,
                        title=f"rollout block {bi} (row {r})", out_path=out,
                        quantiles=qs, mode=args.plot_mode, ghi_cs_row=gcs,
                        primary=modes[0], spaghetti=args.spaghetti,
                        spaghetti_alpha=args.spaghetti_alpha,
                        spaghetti_max=args.spaghetti_max)
        print("  saved", out)


def cmd_compare(args):
    fm, W = _load_model_and_windows(args)
    rows = np.asarray(args.rows, int)
    have_gcs = "fut_ghi_cs" in W
    qs = fm.cfg["eval"].get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])

    dq = None
    if args.deep_quantile_checkpoint:
        dq = load_any(args.deep_quantile_checkpoint, device=core.get_device(
            args.device))
        # a shape mismatch here would otherwise surface as a cryptic tensor
        # broadcast error deep inside torch -- fail fast with the actual cause
        if dq.H_in != fm.H_in or dq.H_out != fm.H_out or dq.K != fm.K:
            raise UnsupportedByCheckpoint(
                f"deep_quantile checkpoint (H_in={dq.H_in}, H_out={dq.H_out}, "
                f"K={dq.K}) doesn't match the flow checkpoint "
                f"(H_in={fm.H_in}, H_out={fm.H_out}, K={fm.K}). These must "
                f"come from the SAME task (history_days/forecast_days) and "
                f"preprocess run to be compared on one chart -- this isn't "
                f"overridable with --allow-unsupported, the arrays literally "
                f"don't line up.")

    baseline_names = (list(BASELINE_CLASSES.keys()) if args.baselines == "all"
                      else ([] if args.baselines == "none"
                            else args.baselines.split(",")))
    baselines = fit_baselines(baseline_names, W, fm.cfg, fm.K, fm.n_days)

    for r in rows:
        fut_nwp = slice_nwp_from_windows(W, np.array([r]))
        gcs = W["fut_ghi_cs"][r:r + 1] if have_gcs else None
        rng = np.random.default_rng(args.seed)
        series = {"flow": fm.predict_ensemble(
            W["hist_csi"][r:r + 1], W["hist_zen"][r:r + 1],
            W["fut_zen"][r:r + 1], W["hist_mask"][r:r + 1],
            W["fut_mask"][r:r + 1], fut_ghi_cs=gcs, fut_nwp=fut_nwp,
            n_ensemble=args.n_ensemble, rng=rng)[0]}
        if dq is not None:
            rng = np.random.default_rng(args.seed)
            series["deep_quantile"] = dq.predict_ensemble(
                W["hist_csi"][r:r + 1], W["hist_zen"][r:r + 1],
                W["fut_zen"][r:r + 1], W["hist_mask"][r:r + 1],
                W["fut_mask"][r:r + 1], fut_ghi_cs=gcs, fut_nwp=fut_nwp,
                n_ensemble=args.n_ensemble, rng=rng)[0]
        for name, b in baselines.items():
            rng = np.random.default_rng(args.seed)
            try:
                series[name] = b.predict_ensemble(
                    W["hist_csi"][r:r + 1], W["hist_zen"][r:r + 1],
                    W["fut_zen"][r:r + 1], W["hist_mask"][r:r + 1],
                    W["fut_mask"][r:r + 1], fut_ghi_cs=gcs, fut_nwp=fut_nwp,
                    n_ensemble=args.n_ensemble, rng=rng)[0]
            except Exception as e:
                warnings.warn(f"{name} predict failed on row {r}: {e}")

        t = W["fut_csi"][r] if "fut_csi" in W else None
        gcs_row = W["fut_ghi_cs"][r] if have_gcs else None
        out = os.path.join(args.out, f"compare_row{r}_{args.mode}.png")
        plot_comparison(series, t, W["fut_mask"][r], fm.K, fm.n_days,
                        title=f"row {r}: flow vs deep_quantile vs baselines",
                        out_path=out, quantiles=qs, mode=args.mode,
                        ghi_cs_row=gcs_row, primary="flow",
                        spaghetti=args.spaghetti,
                        spaghetti_alpha=args.spaghetti_alpha,
                        spaghetti_max=args.spaghetti_max)
        print("  saved", out)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_windows=True):
        p.add_argument("--checkpoint", required=True)
        if need_windows:
            p.add_argument("--windows", required=True)
        p.add_argument("--device", default="auto")
        p.add_argument("--n-ensemble", type=int, default=100)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default="forecasts")
        p.add_argument("--spaghetti", action="store_true",
                       help="overlay individual ensemble member trajectories "
                            "on the fan chart, at low opacity")
        p.add_argument("--spaghetti-alpha", type=float, default=0.05,
                       help="opacity of each spaghetti line (default 0.05)")
        p.add_argument("--spaghetti-max", type=int, default=60,
                       help="max spaghetti lines drawn per series, evenly "
                            "sampled across the ensemble (default 60)")

    pl = sub.add_parser("list", help="discover checkpoints in models_dir")
    pl.add_argument("--models-dir", required=True)
    pl.set_defaults(func=cmd_list)

    pf = sub.add_parser("forecast", help="plain forecast + fan chart")
    common(pf)
    pf.add_argument("--rows", type=int, nargs="+", default=[0])
    pf.add_argument("--mode", choices=["csi", "ghi"], default="ghi")
    pf.set_defaults(func=cmd_forecast)

    pa = sub.add_parser("availability",
                        help="forecast with/without history and/or NWP")
    common(pa)
    pa.add_argument("--rows", type=int, nargs="+", default=[0])
    pa.add_argument("--mode", choices=["csi", "ghi"], default="ghi")
    pa.add_argument("--nwp-fill", choices=["zero", "neutral"], default="zero",
                    help="'zero' matches production fut_nwp=None exactly; "
                         "'neutral' additionally adds a diagnostic variant "
                         "that keeps present=1 but fills climatologically "
                         "neutral content, to isolate whether a wild "
                         "no_nwp result is 'zero reads as an extreme "
                         "signal' vs 'the net can't handle absence at all' "
                         "-- see availability_variants' docstring")
    pa.add_argument("--allow-unsupported", action="store_true",
                    help="compute variants this checkpoint was never "
                         "trained for (e.g. no_nwp with no NWP-dropout "
                         "training) instead of skipping them. Output is a "
                         "labeled diagnostic, not a sound forecast.")
    pa.set_defaults(func=cmd_availability)

    pc = sub.add_parser("coldstart",
                        help="forecast from coordinates + clear-sky only")
    common(pc, need_windows=False)
    pc.add_argument("--lat", type=float, required=True)
    pc.add_argument("--lon", type=float, required=True)
    pc.add_argument("--alt", type=float, default=0.0)
    pc.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    pc.add_argument("--resolution-min", type=float, default=10.0)
    pc.add_argument("--mode", choices=["csi", "ghi"], default="ghi")
    pc.add_argument("--allow-unsupported", action="store_true",
                    help="run cold start even if this checkpoint was never "
                         "trained without history and/or without NWP. "
                         "Output is a labeled diagnostic, not a sound "
                         "forecast.")
    pc.set_defaults(func=cmd_coldstart)

    pr = sub.add_parser("rollout",
                        help="autoregressive vs direct multi-block rollout")
    common(pr)
    pr.add_argument("--start-row", type=int, required=True)
    pr.add_argument("--n-blocks", type=int, default=3)
    pr.add_argument("--mode", choices=["direct", "autoregressive", "both"],
                    default="both")
    pr.add_argument("--plot-mode", choices=["csi", "ghi"], default="ghi")
    pr.add_argument("--rollout-ensemble", action="store_true",
                    help="feed forward all ensemble paths (slower, honest "
                         "compounding) instead of the point median")
    pr.set_defaults(func=cmd_rollout)

    pcp = sub.add_parser("compare",
                         help="flow vs deep_quantile vs classical baselines")
    common(pcp)
    pcp.add_argument("--rows", type=int, nargs="+", default=[0])
    pcp.add_argument("--mode", choices=["csi", "ghi"], default="ghi")
    pcp.add_argument("--deep-quantile-checkpoint", default=None)
    pcp.add_argument("--baselines", default="all",
                     help="'all', 'none', or comma-separated names from "
                          f"{list(BASELINE_CLASSES)}")
    pcp.set_defaults(func=cmd_compare)

    # ps = sub.add_parser("selftest",
    #                     help="exercise every variant against synthetic data")
    # ps.set_defaults(func=lambda args: run_selftest_variants())
    return ap


# # ============================================================================
# # ===== SECTION: self-test ====================================================
# # ============================================================================
# def _tiny_cfg():
#     """A DEFAULT_CONFIG clone shrunk for a fast, real, end-to-end test."""
#     cfg = core.json_roundtrip(core.DEFAULT_CONFIG)
#     cfg["task"]["history_days"] = 2
#     cfg["task"]["forecast_days"] = 1
#     cfg["data"]["min_daylight_steps"] = 12
#     cfg["model"]["hidden"] = 16
#     cfg["model"]["n_blocks"] = 2
#     cfg["model"]["cond_embed_dim"] = 16
#     cfg["model"]["time_embed_dim"] = 16
#     cfg["model"]["attention"] = True
#     cfg["model"]["n_heads"] = 2
#     cfg["model"]["max_blocks"] = 8
#     cfg["model"]["max_dilation"] = 64
#     cfg["train"]["epochs"] = 2
#     cfg["train"]["batch_size"] = 32
#     cfg["train"]["n_sampling_steps"] = 3
#     cfg["train"]["early_stopping"] = False
#     cfg["train"]["calibrate"] = False
#     cfg["train"]["history_dropout"] = 0.2
#     cfg["train"]["device"] = "cpu"
#     cfg["experiment"]["n_ensemble"] = 8
#     cfg["model"]["condition_nwp"] = False
#     cfg["model"]["condition_site"] = False
#     cfg["sites"] = None
#     return cfg


# def run_selftest_variants():
#     print("=== forecast_variants SELF-TEST ===")
#     cfg = _tiny_cfg()
#     rng = np.random.default_rng(0)
#     dates, steps, csis, zens, gcss = core._synthetic_rows(rng, n_days=90)
#     days = core.assemble_days(cfg, dates, steps, csis, zens,
#                               np.ones(len(csis), bool), ghi_cs=gcss)
#     K, n_days_grid = int(days["K"]), 1
#     W = core.build_day_windows(cfg, days)
#     N = W["fut_csi"].shape[0]
#     print(f"K={K} windows={N}")

#     rep = core.make_representation(cfg, "raw").fit(W["fut_csi"][W["fut_mask"]])
#     prior = core.make_prior(cfg, "clearsky", K, cfg["task"]["forecast_days"])
#     fm = core.FlowMatcher(cfg, W["hist_csi"].shape[1], W["fut_csi"].shape[1],
#                           K, cfg["task"]["forecast_days"], prior, rep,
#                           core.get_device("cpu"))
#     tr = np.arange(N // 2)
#     va = np.arange(N // 2, N)
#     fm.prior.fit_for(rep.encode(W["fut_csi"][tr]),
#                      core.self_fill(rep.encode(W["hist_csi"][tr]),
#                                     W["hist_mask"][tr], rep.clearsky_code()),
#                      W["hist_mask"][tr], W["fut_mask"][tr], rep.clearsky_code(),
#                      hist_csi_phys=W["hist_csi"][tr])
#     fm.fit(W["hist_csi"][tr], W["fut_csi"][tr], W["hist_zen"][tr],
#           W["fut_zen"][tr], W["hist_mask"][tr], W["fut_mask"][tr],
#           fut_ghi_cs=W["fut_ghi_cs"][tr], es_split=None, rng=rng)
#     print("A) tiny FlowMatcher trained  OK")

#     # 1) plain forecast
#     pred = fm.predict_ensemble(W["hist_csi"][va], W["hist_zen"][va],
#                                W["fut_zen"][va], W["hist_mask"][va],
#                                W["fut_mask"][va], fut_ghi_cs=W["fut_ghi_cs"][va],
#                                n_ensemble=8, rng=rng)
#     assert np.all(np.isfinite(pred)), "plain forecast has non-finite values"
#     print("B) plain predict_ensemble  OK")

#     # 2) availability variants -- capability-gated by default
#     variants, skipped = availability_variants(fm, W, va[:3], n_ensemble=6,
#                                               seed=1)
#     assert set(variants) == {"full", "no_history"}, variants.keys()
#     assert skipped == {}, skipped  # this tiny cfg has no NWP at all
#     for name, ens in variants.items():
#         assert np.all(np.isfinite(ens)), f"{name} has non-finite values"
#     # full vs no_history should generally differ somewhere
#     assert not np.allclose(variants["full"], variants["no_history"]), \
#         "no_history variant identical to full -- history had no effect"
#     print("C) availability variants (full / no_history)  OK")

#     # 3) cold start (coordinates + clear-sky only)
#     ens, geo = coldstart_forecast(fm, 40.05192, -88.37309, 213.0,
#                                   "2024-06-01", n_ensemble=6, seed=2)
#     assert ens.shape == (1, 6, fm.H_out)
#     assert np.all(np.isfinite(ens)), "coldstart forecast has non-finite values"
#     assert geo["fut_mask"].any(), "coldstart produced an all-night mask"
#     print("D) coldstart (pvlib zenith + Ineichen clear-sky)  OK")

#     # 4) rollout: direct vs autoregressive
#     W["first_day_ord"] = W["first_day_ord"]              # present already
#     n_blocks = 3
#     valid_starts = [i for i in range(N - n_blocks)
#                     if all(np.any(W["first_day_ord"] ==
#                                   W["first_day_ord"][i] + k)
#                           for k in range(n_blocks))]
#     assert valid_starts, "no valid rollout start found in synthetic windows"
#     sr = valid_starts[0]
#     res_d = rollout_forecast(fm, W, sr, n_blocks, mode="direct",
#                              n_ensemble=6, seed=3)
#     res_a = rollout_forecast(fm, W, sr, n_blocks, mode="autoregressive",
#                              n_ensemble=6, seed=3, feedback="median")
#     assert res_d["pred"].shape == (n_blocks, 6, fm.H_out)
#     assert res_a["pred"].shape == (n_blocks, 6, fm.H_out)
#     assert np.all(np.isfinite(res_d["pred"])) and np.all(
#         np.isfinite(res_a["pred"])), "rollout produced non-finite values"
#     res_ae = rollout_forecast(fm, W, sr, n_blocks, mode="autoregressive",
#                               n_ensemble=6, seed=3, feedback="ensemble")
#     assert np.all(np.isfinite(res_ae["pred"]))
#     print("E) rollout direct / autoregressive (point + ensemble feedback)  OK")

#     # 5) DeepQuantile + baselines + comparison plot
#     dq = core.DeepQuantile(cfg, W["hist_csi"].shape[1], W["fut_csi"].shape[1],
#                            K, cfg["task"]["forecast_days"], rep,
#                            core.get_device("cpu"))
#     dq.fit(W["hist_csi"][tr], W["fut_csi"][tr], W["hist_zen"][tr],
#           W["fut_zen"][tr], W["hist_mask"][tr], W["fut_mask"][tr],
#           fut_ghi_cs=W["fut_ghi_cs"][tr], rng=rng, n_quantiles=8)
#     dq_pred = dq.predict_ensemble(W["hist_csi"][va[:1]], W["hist_zen"][va[:1]],
#                                   W["fut_zen"][va[:1]], W["hist_mask"][va[:1]],
#                                   W["fut_mask"][va[:1]],
#                                   fut_ghi_cs=W["fut_ghi_cs"][va[:1]],
#                                   n_ensemble=8, rng=rng)
#     assert np.all(np.isfinite(dq_pred))
#     print("F) tiny DeepQuantile trained + forecast  OK")

#     baselines = fit_baselines(list(BASELINE_CLASSES.keys()), W, cfg, K,
#                               cfg["task"]["forecast_days"], fit_idx=tr)
#     assert "nwp_direct" not in baselines, \
#         "nwp_direct should self-skip: this cfg has no NWP"
#     assert set(baselines) == {"day_persistence", "peen", "ch_peen",
#                               "analog_day"}, baselines.keys()
#     for name, b in baselines.items():
#         bp = b.predict_ensemble(W["hist_csi"][va[:1]], W["hist_zen"][va[:1]],
#                                 W["fut_zen"][va[:1]], W["hist_mask"][va[:1]],
#                                 W["fut_mask"][va[:1]], n_ensemble=6, rng=rng)
#         assert np.all(np.isfinite(bp)), f"baseline {name} non-finite"
#     print("G) all 4 available classical baselines fit + forecast  OK")

#     # 6) comparison plot (flow fan + deep_quantile + baselines overlaid)
#     series = {"flow": pred[0], "deep_quantile": dq_pred[0]}
#     for name, b in baselines.items():
#         series[name] = b.predict_ensemble(
#             W["hist_csi"][va[:1]], W["hist_zen"][va[:1]], W["fut_zen"][va[:1]],
#             W["hist_mask"][va[:1]], W["fut_mask"][va[:1]], n_ensemble=6,
#             rng=rng)[0]
#     out_dir = "/tmp/forecast_variants_selftest"
#     os.makedirs(out_dir, exist_ok=True)
#     out = plot_comparison(series, W["fut_csi"][va[0]], W["fut_mask"][va[0]],
#                           K, cfg["task"]["forecast_days"], "selftest compare",
#                           os.path.join(out_dir, "compare.png"),
#                           quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], mode="ghi",
#                           ghi_cs_row=W["fut_ghi_cs"][va[0]], primary="flow")
#     assert os.path.exists(out) and os.path.getsize(out) > 0
#     print(f"H) comparison fan chart written to {out}  OK")

#     # 7) plot every other variant kind too, just to prove the plotting path
#     # doesn't crash on them (shapes, masks, ghi conversion all differ)
#     av_out = plot_comparison(
#         {k: v[0] for k, v in variants.items()}, W["fut_csi"][va[0]],
#         W["fut_mask"][va[0]], K, cfg["task"]["forecast_days"],
#         "selftest availability", os.path.join(out_dir, "availability.png"),
#         mode="ghi", ghi_cs_row=W["fut_ghi_cs"][va[0]], primary="full")
#     cs_out = plot_forecast(ens[0], None, geo["fut_mask"][0], K, fm.n_days,
#                            "selftest coldstart",
#                            os.path.join(out_dir, "coldstart.png"), mode="ghi",
#                            ghi_cs_row=geo["fut_ghi_cs"][0])
#     ro_out = plot_comparison(
#         {"direct": res_d["pred"][0], "autoregressive": res_a["pred"][0]},
#         res_d["truth"][0] if res_d["truth"] is not None else None,
#         res_d["mask"][0] if res_d["mask"] is not None else
#         np.ones(fm.H_out, bool), K, cfg["task"]["forecast_days"],
#         "selftest rollout", os.path.join(out_dir, "rollout.png"), mode="csi",
#         primary="direct")
#     for p in (av_out, cs_out, ro_out):
#         assert os.path.exists(p) and os.path.getsize(p) > 0
#     print("I) availability / coldstart / rollout fan charts all render  OK")

#     # 8) capability guards actually forbid what a checkpoint can't do
#     guard_cfg = _tiny_cfg()
#     guard_cfg["model"]["condition_nwp"] = True
#     guard_cfg["train"]["history_dropout"] = 0.0     # never trained w/o history
#     guard_cfg["sites"] = None    # DEFAULT_CONFIG ships a 7-site pool; keep
#                                  # this synthetic checkpoint single-site so
#                                  # capability_report's multi_site check means
#                                  # what the test below expects
#     g_rng = np.random.default_rng(7)
#     g_dates, g_steps, g_csis, g_zens, g_gcss = core._synthetic_rows(g_rng,
#                                                                     n_days=90)
#     g_days = core.assemble_days(guard_cfg, g_dates, g_steps, g_csis, g_zens,
#                                 np.ones(len(g_csis), bool), ghi_cs=g_gcss)
#     g_K = int(g_days["K"])
#     g_W = core.build_day_windows(guard_cfg, g_days)
#     gN, gH = g_W["fut_csi"].shape
#     g_noise = g_rng.normal(0, 0.15, size=(gN, gH)).astype(np.float32)
#     g_W["fut_nwp_csi"] = np.clip(g_W["fut_csi"] + g_noise, 0.02, 1.8).astype(
#         np.float32)
#     g_W["fut_nwp_csi_present"] = g_W["fut_mask"].astype(np.float32)
#     g_rep = core.make_representation(guard_cfg, "raw").fit(
#         g_W["fut_csi"][g_W["fut_mask"]])
#     g_prior = core.make_prior(guard_cfg, "clearsky", g_K,
#                               guard_cfg["task"]["forecast_days"])
#     g_fm = core.FlowMatcher(guard_cfg, g_W["hist_csi"].shape[1],
#                             g_W["fut_csi"].shape[1], g_K,
#                             guard_cfg["task"]["forecast_days"], g_prior, g_rep,
#                             core.get_device("cpu"))
#     g_tr = np.arange(gN // 2)
#     g_fut_nwp_tr = slice_nwp_from_windows(g_W, g_tr)
#     g_fm.prior.fit_for(
#         g_rep.encode(g_W["fut_csi"][g_tr]),
#         core.self_fill(g_rep.encode(g_W["hist_csi"][g_tr]),
#                        g_W["hist_mask"][g_tr], g_rep.clearsky_code()),
#         g_W["hist_mask"][g_tr], g_W["fut_mask"][g_tr], g_rep.clearsky_code(),
#         hist_csi_phys=g_W["hist_csi"][g_tr])
#     g_fm.fit(g_W["hist_csi"][g_tr], g_W["fut_csi"][g_tr], g_W["hist_zen"][g_tr],
#             g_W["fut_zen"][g_tr], g_W["hist_mask"][g_tr], g_W["fut_mask"][g_tr],
#             fut_ghi_cs=g_W["fut_ghi_cs"][g_tr], fut_nwp=g_fut_nwp_tr, rng=g_rng)
#     g_va = np.arange(gN // 2, gN)

#     cap = capability_report(g_fm)
#     assert cap == {"history_dropout_trained": False, "has_nwp": True,
#                    "nwp_anchored_prior": False, "nwp_dropout_trained": False,
#                    "site_conditioned": False, "multi_site": False}, cap

#     # this checkpoint has NWP conditioning but a non-NWP-anchored prior
#     # (clearsky), so no_nwp is supported by default even with no
#     # nwp_dropout training and no history_dropout -- only the
#     # history-dependent variants are skipped.
#     g_variants, g_skipped = availability_variants(g_fm, g_W, g_va[:1],
#                                                   n_ensemble=6, seed=1)
#     assert set(g_variants) == {"full", "no_nwp"}, \
#         f"expected 'full'+'no_nwp' by default, got {set(g_variants)}"
#     assert set(g_skipped) == {"no_history", "no_history_no_nwp"}, g_skipped
#     print("J) availability guards SKIP no_history/no_history_no_nwp by "
#          "default (history_dropout=0), but ALLOW no_nwp (non-NWP-anchored "
#          "prior)  OK")

#     with warnings.catch_warnings(record=True) as wlist:
#         warnings.simplefilter("always")
#         g_variants2, g_skipped2 = availability_variants(
#             g_fm, g_W, g_va[:1], n_ensemble=6, seed=1, allow_unsupported=True)
#     assert g_skipped2 == {}
#     assert set(g_variants2) == {"full", "no_history", "no_nwp",
#                                 "no_history_no_nwp"}
#     assert any("PROCEEDING PAST UNSUPPORTED" in str(w.message)
#               for w in wlist), "expected an override warning, got none"
#     print("K) --allow-unsupported computes the still-skipped ones anyway, "
#          "loudly  OK")

#     # coldstart still needs history_dropout regardless of NWP, so this
#     # checkpoint (history_dropout=0) is still refused -- but now for the
#     # HISTORY reason only, not NWP (its prior isn't NWP-anchored).
#     try:
#         coldstart_forecast(g_fm, 40.05192, -88.37309, 213.0, "2024-06-01",
#                            n_ensemble=6, seed=2)
#         raise AssertionError("coldstart should have refused (no "
#                              "history_dropout)")
#     except UnsupportedByCheckpoint:
#         pass
#     with warnings.catch_warnings(record=True) as wlist:
#         warnings.simplefilter("always")
#         cs_ens, cs_geo = coldstart_forecast(
#             g_fm, 40.05192, -88.37309, 213.0, "2024-06-01", n_ensemble=6,
#             seed=2, allow_unsupported=True)
#     assert np.all(np.isfinite(cs_ens))
#     print("L) coldstart refuses by default (history only), runs (as a "
#          "labeled diagnostic) with --allow-unsupported  OK")

#     # ---- M/N: prove the NWP-anchored-prior distinction actually matters --
#     # same synthetic data, history_dropout ON this time, but two prior
#     # kinds: 'nwp' (should stay blocked without --allow-unsupported) vs
#     # 'clearsky' (should be allowed, demonstrating the fix: only a
#     # NWP-anchored prior keeps a checkpoint from cold-starting).
#     def _fit_guard_checkpoint(prior_kind):
#         gcfg = _tiny_cfg()
#         gcfg["model"]["condition_nwp"] = True
#         gcfg["train"]["history_dropout"] = 0.1     # trained WITHOUT history
#         gcfg["sites"] = None
#         rep = core.make_representation(gcfg, "raw").fit(
#             g_W["fut_csi"][g_W["fut_mask"]])
#         prior = core.make_prior(gcfg, prior_kind, g_K,
#                                 gcfg["task"]["forecast_days"])
#         fm_ = core.FlowMatcher(gcfg, g_W["hist_csi"].shape[1],
#                                g_W["fut_csi"].shape[1], g_K,
#                                gcfg["task"]["forecast_days"], prior, rep,
#                                core.get_device("cpu"))
#         nwp_anchor_tr = (core.self_fill(
#             rep.encode(g_W["fut_nwp_csi"][g_tr]), g_W["fut_mask"][g_tr],
#             rep.clearsky_code()) if prior_kind == "nwp" else None)
#         fm_.prior.fit_for(
#             rep.encode(g_W["fut_csi"][g_tr]),
#             core.self_fill(rep.encode(g_W["hist_csi"][g_tr]),
#                            g_W["hist_mask"][g_tr], rep.clearsky_code()),
#             g_W["hist_mask"][g_tr], g_W["fut_mask"][g_tr], rep.clearsky_code(),
#             hist_csi_phys=g_W["hist_csi"][g_tr], fut_nwp_anchor=nwp_anchor_tr)
#         fm_.fit(g_W["hist_csi"][g_tr], g_W["fut_csi"][g_tr],
#                g_W["hist_zen"][g_tr], g_W["fut_zen"][g_tr],
#                g_W["hist_mask"][g_tr], g_W["fut_mask"][g_tr],
#                fut_ghi_cs=g_W["fut_ghi_cs"][g_tr], fut_nwp=g_fut_nwp_tr,
#                rng=np.random.default_rng(11))
#         return fm_

#     fm_nwp = _fit_guard_checkpoint("nwp")
#     cap_nwp = capability_report(fm_nwp)
#     assert cap_nwp["nwp_anchored_prior"] is True
#     assert cap_nwp["history_dropout_trained"] is True
#     try:
#         coldstart_forecast(fm_nwp, 40.05192, -88.37309, 213.0, "2024-06-01",
#                            n_ensemble=6, seed=3)
#         raise AssertionError("coldstart on an NWP-anchored-prior checkpoint "
#                              "should still refuse by default")
#     except UnsupportedByCheckpoint:
#         pass

#     fm_cs = _fit_guard_checkpoint("clearsky")
#     cap_cs = capability_report(fm_cs)
#     assert cap_cs["nwp_anchored_prior"] is False
#     assert cap_cs["history_dropout_trained"] is True
#     cs_ens2, _ = coldstart_forecast(fm_cs, 40.05192, -88.37309, 213.0,
#                                     "2024-06-01", n_ensemble=6, seed=3)
#     assert np.all(np.isfinite(cs_ens2))
#     print("M) a checkpoint trained WITHOUT history (history_dropout>0) "
#          "cold-starts by DEFAULT as long as its prior isn't NWP-anchored; "
#          "an otherwise-identical NWP-anchored-prior checkpoint still "
#          "correctly refuses  OK")

#     print("=== ALL forecast_variants SELF-TESTS PASSED ===")


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)