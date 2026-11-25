"""
Mediation kernel size -> saturation -> Δ for the DiracNet-50 / ResNet-18 learning-rate sweep (lr NOT a predictor),
same models / bootstrap / outputs as learning_rate_sweep_plotting.mediation_analysis_lr_sweep for ResNet-50 / VGG-19.

Data: dirac_resnet18_lr_sweep_pruning_saturation_combined_pr_<pr>.csv (build_dirac_resnet18_lr_sweep_pruning_
saturation_combined.py). Saturation = mean depth-scaled train saturation of the pruned (BN-recalibrated) model.
Preprocessing = load_lr_mediation_df: scaled_delta = delta / dense_acc, Kernel_Size = rf_level + 1, Kernel_Size and
scaled_saturation standardized over BOTH networks before subsetting by model.

Sensitivity: at PR 0.9 ResNet-18 RF1 collapses to chance (10%) for most lrs, so PR 0.9 is repeated without runs
whose pruned accuracy is <= 11% (chance floor bounds Δ).
"""
import numpy as np
import pandas as pd

from learning_rate_sweep_plotting import LR_MEDIATION_OUTCOME_MODELS, lr_mediation_plots
from mediation_dirac_resnet18 import analyse, write_excel

CSV_TEMPLATE = "dirac_resnet18_lr_sweep_pruning_saturation_combined_pr_{}.csv"
CHANCE_CUTOFF = 11.0
PRUNING_RATES = (0.5, 0.6, 0.7, 0.8, 0.9)


def load_df(pr, drop_chance=False):
    df = pd.read_csv(CSV_TEMPLATE.format(pr))
    if drop_chance:
        df = df[df["pruned_acc"] > CHANCE_CUTOFF]
    df = df.rename(columns={"scaled_saturation_mean": "scaled_saturation"})
    df["scaled_delta"] = df["delta"] / df["dense_acc"]
    df["Kernel_Size"] = (df["rf_level"] + 1).astype(float)
    for col in ["Kernel_Size", "scaled_saturation"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df.reset_index(drop=True)


def main(n_boot=2000, save_folder="/home/luisaam/Pictures"):
    summary = []
    for pr in PRUNING_RATES:
        for drop_chance in ((False, True) if pr == 0.9 else (False,)):
            df = load_df(pr, drop_chance)
            for model in ("diracnet50", "resnet18"):
                d = df[df["model"] == model].reset_index(drop=True)
                tag = f"{model}_pr_{pr}" + ("_no_chance" if drop_chance else "")
                r = analyse(d, n_boot)
                write_excel(f"mediation_analysis_dirac_resnet18_lr_sweep_{tag}.xlsx", d, r)
                if not drop_chance:
                    lr_mediation_plots(d, r["bootstrap"], f"{save_folder}/mediation_dirac_resnet18_lr_sweep_{tag}")
                b = r["bootstrap"].set_index(["model", "effect"])
                dec = r["decomposition"]
                ws = r["within_slopes"].set_index("rf_level")
                for om in LR_MEDIATION_OUTCOME_MODELS:
                    row = {"network": model, "pr": pr, "chance runs dropped": drop_chance, "n": len(d),
                           "n chance (pruned <= 11%)": int((d["pruned_acc"] <= CHANCE_CUTOFF).sum()),
                           "outcome model": om, "R2": r["fit"].loc[om, "R2"]}
                    for e in ("NDE", "NIE", "TE", "NIE / TE"):
                        row[e] = b.loc[(om, e), "estimate"]
                        row[f"{e} CI"] = f"[{b.loc[(om, e), 'ci_low']:.3f}, {b.loc[(om, e), 'ci_high']:.3f}]"
                    row["dominance % saturation"] = dec.loc["scaled_saturation", "general dominance % of R2"]
                    row["dominance % kernel"] = dec.loc["Kernel_Size", "general dominance % of R2"]
                    row["saturation unique R2"] = dec.loc["scaled_saturation", "unique R2 (added last)"]
                    row["saturation p added last"] = dec.loc["scaled_saturation", "p added last"]
                    row["mediator R2"] = r["mediator"].rsquared
                    for lvl in ws.index:
                        row[f"within slope RF{lvl}"] = ws.loc[lvl, "slope"]
                        row[f"within p RF{lvl}"] = ws.loc[lvl, "p"]
                        row[f"within R2 RF{lvl}"] = ws.loc[lvl, "R2"]
                    summary.append(row)
                print(f"\n===== {tag} =====")
                print(r["fit"].drop(columns="formula").round(3).to_string())
                print(r["bootstrap"].round(4).to_string())
                print(dec.round(3).to_string())
                print(r["within_slopes"].round(4).to_string())
    summary = pd.DataFrame(summary)
    summary.to_excel("mediation_analysis_dirac_resnet18_lr_sweep_summary.xlsx", index=False)
    return summary


if __name__ == "__main__":
    s = main()
    pd.set_option("display.width", 300)
    print(s.round(3).to_string())
