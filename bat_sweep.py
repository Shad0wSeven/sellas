"""Conditional REGAL Monte Carlo over fixed BAT Weibull OS curves.

Run from regal_model/ with, for example:
    python3 bat_sweep.py --n-sim 100000

By default, BAT_GRID sweeps median OS (months) x 3-year OS. Use `--lock` to
run one chosen pair with zero variance on both BAT knobs. All other priors and
reporting assumptions are inherited from regal_model.py.
"""
import argparse
import csv
import numpy as np

import regal_model as r


# Sensible starter grid; expand or replace for a focused analysis.
BAT_GRID = [
    (8.0, 0.10), (8.0, 0.18), (8.0, 0.27),
    (12.5, 0.10), (12.5, 0.18), (12.5, 0.27),
    (16.0, 0.10), (16.0, 0.18), (16.0, 0.27),
]


def weighted_mean(x, w):
    return float(np.sum(np.asarray(x) * w) / max(np.sum(w), 1e-300))


def weighted_quantiles(x, w, prefix):
    q = r.wq(np.asarray(x), np.asarray(w), np.array([0.10, 0.50, 0.90]))
    return {f"{prefix}_p{p}": float(v) for p, v in zip((10, 50, 90), q)}


def validate_pair(median, s36):
    if not (median > 0 and 0 < s36 < 1 and median != 36):
        raise ValueError(f"Invalid BAT Weibull inputs: median={median}, S36={s36}")
    k = float(r.weib_k(np.array([median]), np.array([s36]))[0])
    if not np.isfinite(k) or k <= 0:
        raise ValueError(
            "A positive-shape Weibull requires S36 < 0.5 when median < 36, "
            "or S36 > 0.5 when median > 36."
        )
    return k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=9106)
    parser.add_argument("--out", default="results/bat_weibull_sweep.csv")
    parser.add_argument("--scenario", default="S0 no lag", choices=sorted(r.SCENARIOS))
    parser.add_argument("--enroll-midpoint", type=float, default=31.0,
                        help="logistic enrollment midpoint, trial-months after 2021-02-08")
    parser.add_argument("--enroll-steepness", type=float, default=0.22,
                        help="logistic enrollment steepness per month")
    parser.add_argument("--bat-pipe-med", type=float, default=None,
                        help="BAT pipeline-delay median in months; defaults to scenario value")
    parser.add_argument("--gps-pipe-med", type=float, default=None,
                        help="GPS pipeline-delay median in months; defaults to scenario value")
    parser.add_argument("--lock", action="store_true",
                        help="run one BAT scenario with BAT median OS and 3-year OS fixed (zero variance)")
    parser.add_argument("--bat-median", type=float, default=12.5,
                        help="BAT median OS in months when --lock is set (default: 12.5)")
    parser.add_argument("--bat-os-36", type=float, default=0.18,
                        help="BAT 3-year OS probability when --lock is set (default: 0.18)")
    args = parser.parse_args()

    scenario = dict(r.SCENARIOS[args.scenario])
    scenario["enrollment"] = dict(family="logistic", midpoint=args.enroll_midpoint,
                                  steepness=args.enroll_steepness)
    if args.bat_pipe_med is not None:
        scenario["pipe_med_bat"] = args.bat_pipe_med
    if args.gps_pipe_med is not None:
        scenario["pipe_med_gps"] = args.gps_pipe_med
    bat_pipe_med = scenario.get("pipe_med_bat", scenario.get("pipe_med", 0.0))
    gps_pipe_med = scenario.get("pipe_med_gps", scenario.get("pipe_med", 0.0))

    rows = []
    bat_cases = [(args.bat_median, args.bat_os_36)] if args.lock else BAT_GRID
    for j, (median, s36) in enumerate(bat_cases):
        k = validate_pair(median, s36)
        prior = dict(
            M=(median, median), S36=(s36, s36),
            c=(0.0, 0.65), theta=(0.3, 1.0), L=(1.0, 8.0),
            leak=(0.002, 0.009), gamma=(1.9, 1.9),
        )
        res = r.run_scenario(
            f"BAT M={median:g}, S36={s36:g}", scenario,
            n_sim=args.n_sim, seed=args.seed + j, prior=prior,
        )
        w = res["w"]
        interim_success = res["z_int"] <= -r.Z1
        continued = ~interim_success
        wc = w * continued
        final_success = res["z_final"] <= -r.ZF
        sequential_success = interim_success | (continued & final_success)
        rows.append({
            "bat_median_months": median,
            "bat_os_36m": s36,
            "bat_median_variance": 0.0,
            "bat_os_36m_variance": 0.0,
            "bat_locked": args.lock,
            "weibull_shape": k,
            "scenario": args.scenario,
            "bat_pipeline_delay_median_months": bat_pipe_med,
            "gps_pipeline_delay_median_months": gps_pipe_med,
            "enrollment_family": "truncated_logistic_early_plus_uniform_backfill",
            "enrollment_midpoint_trial_month": args.enroll_midpoint,
            "enrollment_steepness_per_month": args.enroll_steepness,
            "simulations": args.n_sim,
            "effective_samples_fit": r.ess(w),
            "p_interim_no_efficacy_stop": weighted_mean(continued, w),
            "p_success_overall": weighted_mean(sequential_success, w),
            "gps_os_36m_weighted": weighted_mean(res["gps_S36"], w),
            "hr_at_80th_weighted": weighted_mean(res["hr_final"], wc),
            "p_success_at_80th_given_interim_continuation": weighted_mean(
                res["z_final"] <= -r.ZF, wc
            ),
        })
        # Show the range of GPS assumptions and Cox HRs compatible with the
        # locked BAT curve and the public pooled event-count constraints.
        rows[-1].update(weighted_quantiles(res["c"], w, "gps_c_durable_fraction"))
        rows[-1].update(weighted_quantiles(res["theta"], w, "gps_theta_non_durable_HR"))
        rows[-1].update(weighted_quantiles(res["L"], w, "gps_onset_months"))
        rows[-1].update(weighted_quantiles(res["leak"], w, "gps_durable_leak"))
        rows[-1].update(weighted_quantiles(res["gps_S36"], w, "gps_os_36m"))
        rows[-1].update(weighted_quantiles(res["hr_final"], wc, "cox_hr_at_80th_given_continuation"))
        print(rows[-1], flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} BAT scenarios to {args.out}")


if __name__ == "__main__":
    main()
