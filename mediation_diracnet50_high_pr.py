"""
DiracNet-50 learning-rate-sweep mediation (kernel size -> saturation -> scaled Δ, lr NOT a predictor) at pruning rates
0.7, 0.8, 0.9 and the high rates 0.95, 0.97, 0.99, plus the link to the identity-pruning measurements.

Data: dirac_resnet18_lr_sweep_pruning_saturation_combined_pr_<pr>.csv (diracnet50 rows; 0.95-0.99 built from the
slurm_diracnet50_lr_sweep_high_pr_pruning_handler.sh results) and dirac_identity_pruning_results/*summary*.csv
(measure_dirac_identity_pruning.py).

Kernel size and saturation are standardized over DiracNet-50 alone (ResNet-18 has no high-rate runs). NDE / NIE / TE,
R² and dominance shares are invariant to that choice (linear rescaling of x and y), so the 0.7-0.9 numbers equal those
of mediation_lr_sweep_all_models.py. Rows with a missing saturation file are dropped.

Variants per pruning rate:  all runs;  no_chance (pruned accuracy > 11%, only if any run is at chance);
trained (lr >= 0.003: drops the undertrained lr 0.0001 / 0.001 runs, dense accuracy 28% / 72%).

Outputs: mediation_analysis_diracnet50_high_pr_pr_<pr>[_<variant>].xlsx, mediation_analysis_diracnet50_high_pr_summary.xlsx,
statistical_analysis_saturation_prunability/mediation_diracnet50_high_pr_results.json and
statistical_analysis_saturation_prunability/images/mediation_diracnet50_high_pr_pr_<pr>_*.pdf
"""
import glob
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from learning_rate_sweep_plotting import LR_MEDIATION_OUTCOME_MODELS, lr_mediation_plots
from mediation_dirac_resnet18 import analyse, write_excel
from mediation_lr_sweep_all_models import to_json_result

CSV = "dirac_resnet18_lr_sweep_pruning_saturation_combined_pr_{}.csv"
IDENTITY_GLOB = "dirac_identity_pruning_results/diracnet50_identity_pruning_summary_rf_*.csv"
PRUNING_RATES = ("0.7", "0.8", "0.9", "0.95", "0.97", "0.99")
CHANCE_CUTOFF = 11.0
OUT_DIR = "statistical_analysis_saturation_prunability"

# Which saturation is the mediator. "pruned" (default): saturation of the pruned, BN-recalibrated model
# (scaled_saturation_mean); "dense": saturation of the dense model at the checkpoint epoch (dense_saturation_mean, the
# same for every pruning rate). Dense outputs go to statistical_analysis_saturation_prunability/dense_saturation_mediation/
# with "_dense_saturation" in every file name.
SAT_COL, XLSX_DIR, SUFFIX = "scaled_saturation_mean", ".", ""


def set_mode(saturation):
    global SAT_COL, XLSX_DIR, OUT_DIR, SUFFIX
    if saturation == "dense":
        SAT_COL, SUFFIX = "dense_saturation_mean", "_dense_saturation"
        OUT_DIR = XLSX_DIR = "statistical_analysis_saturation_prunability/dense_saturation_mediation"
        os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
    elif saturation != "pruned":
        raise ValueError(saturation)


def load_df(pr, variant="all"):
    df = pd.read_csv(CSV.format(pr))
    df = df[(df["model"] == "diracnet50") & df[SAT_COL].notna()]
    if variant == "no_chance":
        df = df[df["pruned_acc"] > CHANCE_CUTOFF]
    if variant == "trained":
        df = df[df["lr"] >= 0.003]
    df = df.copy()
    df["scaled_saturation"] = df[SAT_COL]
    df["saturation_raw"] = df["scaled_saturation"]
    df["scaled_delta"] = df["delta"] / df["dense_acc"]
    df["Kernel_Size"] = (df["rf_level"] + 1).astype(float)
    for col in ["Kernel_Size", "scaled_saturation"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df.reset_index(drop=True)


def run_job(job):
    pr, variant, n_boot = job
    d = load_df(pr, variant)
    tag = f"pr_{pr}" + ("" if variant == "all" else f"_{variant}")
    r = analyse(d, n_boot)
    write_excel(f"{XLSX_DIR}/mediation_analysis_diracnet50_high_pr_{tag}{SUFFIX}.xlsx", d, r)
    res = to_json_result(d, r)
    res["mean_delta"] = float(d["scaled_delta"].mean())
    res["delta_sd_within"] = float(d.groupby("rf_level")["scaled_delta"].std().mean())
    res["mean_pruned_acc_by_rf"] = {int(k): float(v) for k, v in d.groupby("rf_level")["pruned_acc"].mean().items()}
    if variant == "all":
        lr_mediation_plots(d, r["bootstrap"], f"{OUT_DIR}/images/mediation_diracnet50_high_pr_{tag}{SUFFIX}")
        g = d.groupby(["rf_level", "lr"])[["scaled_delta", "saturation_raw", "pruned_acc"]].mean()
        res["by_lr"] = {int(k): {"lr": [float(x) for x in x_.index.get_level_values(1)],
                                 "sd": [round(float(v), 4) for v in x_["scaled_delta"]],
                                 "sat": [round(float(v), 4) for v in x_["saturation_raw"]],
                                 "pruned_acc": [round(float(v), 2) for v in x_["pruned_acc"]]}
                        for k, x_ in g.groupby(level=0)}
    rows = []
    for om in LR_MEDIATION_OUTCOME_MODELS:
        row = {"pr": float(pr), "variant": variant, "n": len(d), "n chance": sum(res["chance_by_rf"].values()),
               "outcome model": om, "R2": res["R2"][om], "mediator R2": res["R2"]["mediator"]}
        for e in ("NDE", "NIE", "TE", "NIE / TE"):
            x = res["effects"][om][e]
            row[e], row[f"{e} CI"] = x["est"], f"[{x['lo']:.3f}, {x['hi']:.3f}]"
        row["dominance % saturation"] = res["dominance"]["scaled_saturation"]["pct"]
        for lvl, w in res["within"].items():
            row[f"within slope RF{lvl}"], row[f"within p RF{lvl}"] = w["slope"], w["p"]
        rows.append(row)
    e = res["effects"]
    line = (f"{tag:22s} n={len(d):3d} chance={sum(res['chance_by_rf'].values()):3d} meanΔ={res['mean_delta']:.3f} "
            f"medR2={res['R2']['mediator']:.2f} TE={e['additive']['TE']['est']:+.3f} "
            f"NIE add={e['additive']['NIE']['est']:+.3f} int={e['interaction']['NIE']['est']:+.3f} "
            f"x2={e['interaction + x^2']['NIE']['est']:+.3f} dom_sat={res['dominance']['scaled_saturation']['pct']:.0f}% "
            "within " + " ".join(f"{w['slope']:+.2f}{'*' if w['p'] < 0.05 else ''}" for w in res["within"].values()))
    return pr, variant, res, rows, line


def identity_link():
    """Per checkpoint and pruning rate: identity-pruning measurements joined to pruned accuracy and saturation."""
    ident = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(IDENTITY_GLOB))], ignore_index=True)
    out = {}
    for pr in PRUNING_RATES:
        d = pd.read_csv(CSV.format(pr))
        d = d[(d["model"] == "diracnet50") & d[SAT_COL].notna()].copy()
        d["scaled_delta"] = d["delta"] / d["dense_acc"]
        idp = ident[np.isclose(ident["pruning_rate"], float(pr))]
        m = d.merge(idp, left_on="name", right_on="checkpoint", suffixes=("", "_id"))
        m_tr = m[m["lr"] >= 0.003]
        corr = lambda x, a, b: float(x[a].corr(x[b]))
        out[pr] = {"n_matched": len(m), "n_rows": len(d),
                   "kept_frac_identity_taps_min": float(m["kept_frac_identity_taps"].min()),
                   "learned_kept_mean": float(m["kept_frac_learned_only"].mean()),
                   "learned_energy_kept_mean": float(m["learned_energy_kept"].mean()),
                   "learned_energy_kept_by_rf": {int(k): float(v) for k, v in m.groupby("rf_level")["learned_energy_kept"].mean().items()},
                   "identity_share_kept_mean": float(m["identity_share_kept"].mean()),
                   "cf_identity_mean": float(m["cf_kept_frac_identity_taps"].mean()),
                   "corr_delta_learned_energy_kept": corr(m, "scaled_delta", "learned_energy_kept"),
                   "corr_delta_learned_energy_kept_trained": corr(m_tr, "scaled_delta", "learned_energy_kept"),
                   "corr_sat_learned_energy_kept": corr(m, SAT_COL, "learned_energy_kept"),
                   "corr_delta_identity_share_dense": corr(m, "scaled_delta", "identity_share_dense"),
                   "corr_delta_identity_share_dense_trained": corr(m_tr, "scaled_delta", "identity_share_dense")}
        # Δ explained by kernel size plus saturation vs plus learned energy kept, and within-RF correlations
        m = m.assign(lek=m["learned_energy_kept"], ks=m["rf_level"] + 1.0, sat=m[SAT_COL])
        r2 = lambda formula: float(smf.ols(formula, m).fit().rsquared)
        out[pr].update({"within_corr_delta_le": [float(x["scaled_delta"].corr(x["lek"])) for _, x in m.groupby("rf_level")],
                        "within_corr_sat_le": [float(x["sat"].corr(x["lek"])) for _, x in m.groupby("rf_level")],
                        "R2_ks": r2("scaled_delta ~ ks"), "R2_ks_sat": r2("scaled_delta ~ ks + sat"),
                        "R2_ks_le": r2("scaled_delta ~ ks + lek"), "R2_ks_sat_le": r2("scaled_delta ~ ks + sat + lek")})
    return out


def main(n_boot=2000, processes=6):
    import multiprocessing as mp
    jobs = []
    for pr in PRUNING_RATES:
        d = pd.read_csv(CSV.format(pr))
        d = d[d["model"] == "diracnet50"]
        variants = ["all", "trained"] + (["no_chance"] if (d["pruned_acc"] <= CHANCE_CUTOFF).any() else [])
        jobs += [(pr, v, n_boot) for v in variants]
    results, summary = {}, []
    with mp.get_context("fork").Pool(processes) as pool:
        for pr, variant, res, rows, line in pool.imap(run_job, jobs):
            results.setdefault(pr, {})[variant] = res
            summary += rows
            print(line, flush=True)
    link = identity_link()
    for pr, v in link.items():
        print(f"identity link PR {pr}: matched {v['n_matched']}/{v['n_rows']}, identity taps kept ≥ {v['kept_frac_identity_taps_min']:.3f}, "
              f"learned energy kept {v['learned_energy_kept_mean']:.3f}, corr(Δ, learned energy kept) {v['corr_delta_learned_energy_kept']:+.2f} "
              f"(trained {v['corr_delta_learned_energy_kept_trained']:+.2f}), corr(sat, learned energy kept) {v['corr_sat_learned_energy_kept']:+.2f}")
    pd.DataFrame(summary).sort_values(["pr", "variant", "outcome model"]).to_excel(
        f"{XLSX_DIR}/mediation_analysis_diracnet50_high_pr_summary{SUFFIX}.xlsx", index=False)
    with open(f"{OUT_DIR}/mediation_diracnet50_high_pr{SUFFIX}_results.json", "w") as f:
        json.dump({"mediation": results, "identity_link": link}, f, indent=1)
    return results, link


if __name__ == "__main__":
    import sys
    # python mediation_diracnet50_high_pr.py [n_boot] [pruned|dense]
    set_mode(sys.argv[2] if len(sys.argv) > 2 else "pruned")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
