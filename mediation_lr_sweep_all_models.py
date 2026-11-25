"""
LR-sweep mediation (kernel size -> saturation -> scaled Δ, lr NOT a predictor) for all four networks at pruning rates
0.7, 0.8 and 0.9, RF levels 1-4.

Pairs and data (standardization of Kernel_Size and saturation is over BOTH networks of a pair, as before):
  resnet50 / vgg19       learning_rate_sweep_pruning_saturation_combined_pr_<pr>.csv
  diracnet50 / resnet18  dirac_resnet18_lr_sweep_pruning_saturation_combined_pr_<pr>.csv
Each (pr, pair) is also run without chance-level runs (pruned accuracy <= 11%), dropped before standardizing.

Contrast for NDE / NIE / TE: smallest -> largest observed kernel size (RF 1 -> RF 4). Same models, bootstrap
(2000 reps stratified by kernel size), dominance and within-kernel slopes as mediation_dirac_resnet18.analyse.

Outputs (same names as the earlier per-pair scripts, so they are overwritten with the RF 1-4 results):
  mediation_analysis_lr_sweep_{resnet50,vgg19}_pr_<pr>[_no_chance].xlsx
  mediation_analysis_dirac_resnet18_lr_sweep_{diracnet50,resnet18}_pr_<pr>[_no_chance].xlsx
  mediation_analysis_lr_sweep_all_models_summary.xlsx
  statistical_analysis_saturation_prunability/mediation_lr_sweep_all_models_results.json  (feeds the decks)
  statistical_analysis_saturation_prunability/images/mediation_{lr_sweep,dirac_resnet18_lr_sweep}_<model>_pr_<pr>_*.pdf
"""
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys

import numpy as np
import pandas as pd

from learning_rate_sweep_plotting import LR_MEDIATION_OUTCOME_MODELS, lr_mediation_plots
from mediation_dirac_resnet18 import analyse, write_excel

CHANCE_CUTOFF = 11.0
PRUNING_RATES = (0.7, 0.8, 0.9)
PAIRS = {
    "resnet50_vgg19": {"csv": "learning_rate_sweep_pruning_saturation_combined_pr_{}.csv",
                       "models": ("resnet50", "vgg19"), "prefix": "mediation_analysis_lr_sweep",
                       "plot_prefix": "mediation_lr_sweep"},
    "dirac_resnet18": {"csv": "dirac_resnet18_lr_sweep_pruning_saturation_combined_pr_{}.csv",
                       "models": ("diracnet50", "resnet18"), "prefix": "mediation_analysis_dirac_resnet18_lr_sweep",
                       "plot_prefix": "mediation_dirac_resnet18_lr_sweep"},
}
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


def load_df(csv, pr, drop_chance=False):
    df = pd.read_csv(csv.format(pr))
    if drop_chance:
        df = df[df["pruned_acc"] > CHANCE_CUTOFF]
    df = df[df[SAT_COL].notna()].copy()
    df["scaled_saturation"] = df[SAT_COL]
    df["saturation_raw"] = df["scaled_saturation"]
    df["scaled_delta"] = df["delta"] / df["dense_acc"]
    df["Kernel_Size"] = (df["rf_level"] + 1).astype(float)
    for col in ["Kernel_Size", "scaled_saturation"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df.reset_index(drop=True)


def to_json_result(d, r):
    b = r["bootstrap"]
    dec = r["decomposition"]
    inter, inter_hc3 = r["outcomes"]["interaction"], r["outcomes_hc3"]["interaction"]
    out = {
        "n": len(d),
        "n_by_rf": {int(k): int(v) for k, v in d["rf_level"].value_counts().sort_index().items()},
        "chance_by_rf": {int(k): int(v) for k, v in
                         d.groupby("rf_level")["pruned_acc"].apply(lambda s: (s <= CHANCE_CUTOFF).sum()).items()},
        "R2": {k: float(v) for k, v in r["fit"]["R2"].items()},
        "effects": {m: {e: {"est": float(row["estimate"]), "lo": float(row["ci_low"]), "hi": float(row["ci_high"])}
                        for e, row in b[b["model"] == m].set_index("effect").iterrows()}
                    for m in LR_MEDIATION_OUTCOME_MODELS},
        "dominance": {p: {"alone": float(dec.loc[p, "R2 alone"]), "unique": float(dec.loc[p, "unique R2 (added last)"]),
                          "pct": float(dec.loc[p, "general dominance % of R2"]),
                          "p_last": float(dec.loc[p, "p added last"])} for p in dec.index},
        "R2_both": float(dec["R2 both"].iloc[0]),
        "within": {int(row["rf_level"]): {"slope": float(row["slope"]), "p": float(row["p"]), "R2": float(row["R2"]),
                                          "n": int(row["n"]), "lo": float(row["ci_low"]), "hi": float(row["ci_high"])}
                   for _, row in r["within_slopes"].iterrows()},
        "interaction": {t: {"coef": float(inter.params[t]), "p": float(inter.pvalues[t]),
                            "p_hc3": float(inter_hc3.pvalues[t])} for t in inter.params.index},
        "sat_range_by_rf": {int(k): [float(g.min()), float(g.max())] for k, g in
                            d.groupby(["rf_level", "lr"])["saturation_raw"].mean().groupby(level=0)},
    }
    return out


def run_job(job):
    pr, pair, drop_chance, n_boot = job
    cfg = PAIRS[pair]
    df = load_df(cfg["csv"], pr, drop_chance)
    out, summary, lines = [], [], []
    for model in cfg["models"]:
        d = df[df["model"] == model].reset_index(drop=True)
        tag = f"{model}_pr_{pr}" + ("_no_chance" if drop_chance else "")
        r = analyse(d, n_boot)
        write_excel(f"{XLSX_DIR}/{cfg['prefix']}_{tag}{SUFFIX}.xlsx", d, r)
        if not drop_chance:
            lr_mediation_plots(d, r["bootstrap"], f"{OUT_DIR}/images/{cfg['plot_prefix']}_{tag}{SUFFIX}")
        res = to_json_result(d, r)
        if not drop_chance:
            g = d.groupby(["rf_level", "lr"])[["scaled_delta", "saturation_raw"]].mean()
            res["by_lr"] = {int(k): {"lr": [float(x) for x in x_.index.get_level_values(1)],
                                     "sd": [round(float(v), 4) for v in x_["scaled_delta"]],
                                     "sat": [round(float(v), 4) for v in x_["saturation_raw"]]}
                            for k, x_ in g.groupby(level=0)}
        out.append((model, str(pr), "no_chance" if drop_chance else "all", res))
        for om in LR_MEDIATION_OUTCOME_MODELS:
            row = {"network": model, "pr": pr, "chance runs dropped": drop_chance, "n": len(d),
                   "n chance": sum(res["chance_by_rf"].values()), "outcome model": om,
                   "R2": res["R2"][om], "mediator R2": res["R2"]["mediator"]}
            for e in ("NDE", "NIE", "TE", "NIE / TE"):
                x = res["effects"][om][e]
                row[e], row[f"{e} CI"] = x["est"], f"[{x['lo']:.3f}, {x['hi']:.3f}]"
            row["dominance % saturation"] = res["dominance"]["scaled_saturation"]["pct"]
            row["dominance % kernel"] = res["dominance"]["Kernel_Size"]["pct"]
            for lvl, w in res["within"].items():
                row[f"within slope RF{lvl}"], row[f"within p RF{lvl}"] = w["slope"], w["p"]
            summary.append(row)
        e = res["effects"]
        lines.append(f"{tag:32s} n={len(d):3d} chance={sum(res['chance_by_rf'].values()):3d} "
                     f"medR2={res['R2']['mediator']:.2f} TE={e['additive']['TE']['est']:+.3f} "
                     f"NIE/TE add={e['additive']['NIE / TE']['est']:+.2f} "
                     f"int={e['interaction']['NIE / TE']['est']:+.2f} "
                     f"x2={e['interaction + x^2']['NIE / TE']['est']:+.2f} "
                     f"dom_sat={res['dominance']['scaled_saturation']['pct']:.0f}%")
    return out, summary, lines


def main(n_boot=2000, processes=6):
    import multiprocessing as mp
    jobs = [(pr, pair, dc, n_boot) for pr in PRUNING_RATES for pair in PAIRS for dc in (False, True)]
    results, summary = {}, []
    with mp.get_context("fork").Pool(processes) as pool:
        for out, rows, lines in pool.imap(run_job, jobs):
            for model, pr, key, res in out:
                results.setdefault(model, {}).setdefault(pr, {})[key] = res
            summary += rows
            print("\n".join(lines), flush=True)
    pd.DataFrame(summary).to_excel(f"{XLSX_DIR}/mediation_analysis_lr_sweep_all_models_summary{SUFFIX}.xlsx", index=False)
    with open(f"{OUT_DIR}/mediation_lr_sweep_all_models{SUFFIX}_results.json", "w") as f:
        json.dump(results, f, indent=1)
    return results


if __name__ == "__main__":
    # python mediation_lr_sweep_all_models.py [n_boot] [pruned|dense]
    set_mode(sys.argv[2] if len(sys.argv) > 2 else "pruned")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
