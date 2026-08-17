# ===== SECTION: main ========================================================
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Day-ahead CSI forecasting with conditional flow matching")
    ap.add_argument("command", choices=["selftest", "preprocess", "run"],
                    help="selftest (no data needed) | preprocess | run")
    ap.add_argument("--quick", action="store_true",
                    help="single purged holdout fold instead of CV")
    ap.add_argument("--final", action="store_true",
                    help="after CV scoring, fit ONE deployable model per "
                         "(rep,prior) on ALL windows and save as _final.pt")
    ap.add_argument("--config", default=None,
                    help="YAML file merged over the default config")
    args = ap.parse_args()

    cfg = json_roundtrip(DEFAULT_CONFIG)
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = _deep_update(cfg, yaml.safe_load(f) or {})

    if args.command == "selftest":
        run_selftest()
        return
    if args.command == "preprocess":
        preprocess(cfg)
        return
    if args.command == "run":
        res = run_experiments(cfg, quick=args.quick)
        os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
        out = os.path.join(cfg["paths"]["results_dir"],
                           f"experiments_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2, default=_json_default)
        print("\nsaved:", out)
        _print_table(res["aggregate"], cfg)
        if args.final:
            saved = train_final(cfg)
            print("\nfinal deployable models:")
            for tag, p in saved.items():
                print(f"  {tag}: {p}")


def _deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _print_table(agg, cfg):

    ref = cfg["eval"].get("dm_reference", "ch_peen")
    n_cfg_folds = int(cfg["split"].get("n_folds", 1))
    methods = sorted(agg, key=lambda k: agg[k].get("crps", float("inf")))
    w = max(len(m) for m in agg) + 1

    def cell(m, k, prec=4):
        v = agg[m].get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return f"{'n/a':>12s}"
        return f"{v:12.{prec}f}"

    groups = [
        ("MARGINAL (per-step)  -- what a quantile head optimizes; expect a tie",
         ["crps", "rmse", "mae", "calibration_err", "coverage_80",
          "enh_frac_pred", "enh_frac_true", "enh_frac_err"]),
        ("GHI-WEIGHTED  -- what the errors cost in energy",
         ["crps_ghi_weighted", "rmse_ghi"]),
        ("JOINT / MULTIVARIATE  -- the flow's home turf (lower=better)",
         ["energy_score", "variogram_score"]),
        ("OPERATOR FUNCTIONALS  -- what a day-ahead decision consumes",
         ["daytotal_crps", "daytotal_coverage_80", "daytotal_energy_crps",
          "ramp_crps"]),
        (f"SKILL & SIGNIFICANCE  (skill>0 = better; vs '{ref}', vs NWP, vs deepq)",
         ["crps_skill_vs_ref", "dm_p_vs_ref", "crps_skill_vs_nwp",
          "dm_p_vs_nwp", "crps_skill_vs_deepq", "vs_skill_vs_deepq",
          "ramp_skill_vs_deepq", "daytotal_skill_vs_deepq"]),
    ]

    print(f"\n{'='*78}\nRESULTS  (mean over folds; coverage targets 0.80; "
          f"lower is better for scores)\n{'='*78}")
    # single-fold / partial-CV warning
    folds_seen = max((agg[m].get("n_folds_scored", 1) for m in agg), default=1)
    if folds_seen < n_cfg_folds:
        print(f"  !! PARTIAL CV: {folds_seen}/{n_cfg_folds} folds scored. "
              f"Treat margins as PRELIMINARY -- joint-skill claims need all "
              f"{n_cfg_folds} folds to be stable.")

    for title, cols in groups:
        # skip a group entirely if no method has any of its metrics
        if not any(c in agg[m] for m in agg for c in cols):
            continue
        print(f"\n-- {title}")
        print(f"{'method':{w}s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
        for m in methods:
            prec = 4 if "score" not in title.split()[0].lower() else 5
            print(f"{m:{w}s} " + " ".join(cell(m, c) for c in cols))

    # the one-line headline the project is about
    if "flow" in " ".join(methods) and "deep_quantile" in agg:
        fl = next((m for m in methods if m.startswith("flow_")), None)
        if fl:
            def g(k):
                v = agg[fl].get(k)
                return "n/a" if v is None or not np.isfinite(v) else f"{v:+.1%}"
            print(f"\n-- HEADLINE: {fl} vs deep_quantile")
            print(f"     marginal CRPS skill : {g('crps_skill_vs_deepq')}   "
                  f"(expect ~tie or slightly negative)")
            print(f"     variogram   skill : {g('vs_skill_vs_deepq')}   "
                  f"(dependence structure -- flow should win)")
            print(f"     ramp        skill : {g('ramp_skill_vs_deepq')}   "
                  f"(joint ramps -- flow should win)")
            print(f"     day-total   skill : {g('daytotal_skill_vs_deepq')}   "
                  f"(coherent day totals -- flow should win)")
    print()


if __name__ == "__main__":
    main()