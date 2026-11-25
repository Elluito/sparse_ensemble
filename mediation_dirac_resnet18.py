"""
Mediation analysis kernel size -> saturation -> Δ for the DiracNet-50 / ResNet-18 control experiments.

Same models and bootstrap as learning_rate_sweep_plotting.mediation_analysis_lr_sweep (VanderWeele NDE / NIE / TE,
2000 bootstrap reps stratified by kernel size, contrast smallest -> largest kernel size, general dominance and
within-kernel-size slopes), applied to dirac_resnet18_pruning_saturation_per_seed_combined.csv.

Preprocessing mirrors load_lr_mediation_df: scaled_delta = delta / dense_acc, Kernel_Size = rf_level + 1, and
Kernel_Size / saturation standardized over BOTH networks (per pruning rate) before subsetting by model.
Saturation = depth-scaled train saturation at the checkpoint epoch (avg_depth_scaled_train_saturation).
Saturation varies within a kernel size only across seeds (no lr sweep here).

Sensitivity run: ResNet-18 RF1 at PR 0.9 has one seed at chance (11.2%), so the analysis is repeated without
runs whose pruned accuracy is <= 15%.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from learning_rate_sweep_plotting import (LR_MEDIATION_MEDIATOR_FORMULA, LR_MEDIATION_OUTCOME_MODELS, lr_ols_table,
                                          lr_mediation_bootstrap, lr_variance_decomposition, lr_within_kernel_slopes,
                                          lr_mediation_plots)

CSV = "dirac_resnet18_pruning_saturation_per_seed_combined.csv"
CHANCE_CUTOFF = 15.0


def load_df(pr, drop_chance=False):
    df = pd.read_csv(CSV)
    df = df[np.isclose(df["pruning_rate"], pr)].copy()
    if drop_chance:
        df = df[df["pruned_acc"] > CHANCE_CUTOFF]
    df = df.rename(columns={"avg_depth_scaled_train_saturation": "scaled_saturation"})
    df["scaled_delta"] = df["delta"] / df["dense_acc"]
    df["Kernel_Size"] = (df["rf_level"] + 1).astype(float)
    for col in ["Kernel_Size", "scaled_saturation"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df.reset_index(drop=True)


def analyse(d, n_boot):
    mediator = smf.ols(LR_MEDIATION_MEDIATOR_FORMULA, d).fit()
    total = smf.ols("scaled_delta ~ Kernel_Size", d).fit()
    outcomes = {name: smf.ols(f, d).fit() for name, f in LR_MEDIATION_OUTCOME_MODELS.items()}
    outcomes_hc3 = {name: smf.ols(f, d).fit(cov_type="HC3") for name, f in LR_MEDIATION_OUTCOME_MODELS.items()}
    fit = pd.DataFrame([{"model": n, "formula": LR_MEDIATION_OUTCOME_MODELS[n], "R2": m.rsquared,
                         "adj R2": m.rsquared_adj, "AIC": m.aic, "BIC": m.bic} for n, m in outcomes.items()] +
                       [{"model": "total (no saturation)", "formula": "scaled_delta ~ Kernel_Size",
                         "R2": total.rsquared, "adj R2": total.rsquared_adj, "AIC": total.aic, "BIC": total.bic},
                        {"model": "mediator", "formula": LR_MEDIATION_MEDIATOR_FORMULA, "R2": mediator.rsquared,
                         "adj R2": mediator.rsquared_adj, "AIC": mediator.aic, "BIC": mediator.bic}]).set_index("model")
    boot = []
    for name, formula in LR_MEDIATION_OUTCOME_MODELS.items():
        t = lr_mediation_bootstrap(d, formula, n_boot)
        t.insert(0, "model", name)
        boot.append(t)
    return {"mediator": mediator, "total": total, "outcomes": outcomes, "outcomes_hc3": outcomes_hc3, "fit": fit,
            "bootstrap": pd.concat(boot, ignore_index=True), "decomposition": lr_variance_decomposition(d),
            "within_slopes": lr_within_kernel_slopes(d)}


def write_excel(path, d, r):
    checks = pd.DataFrame([("n", len(d))] +
                          [(f"n rf_level={k}", v) for k, v in d["rf_level"].value_counts().sort_index().items()] +
                          [(f"{s} {c}", getattr(d[c], s)()) for c in ["Kernel_Size", "scaled_saturation",
                                                                       "scaled_delta"] for s in ["mean", "std"]],
                          columns=["check", "value"]).set_index("check")
    with pd.ExcelWriter(path) as writer:
        checks.to_excel(writer, sheet_name="Data checks")
        r["fit"].to_excel(writer, sheet_name="Model fit")
        lr_ols_table(r["mediator"]).to_excel(writer, sheet_name="Mediator")
        lr_ols_table(r["total"]).to_excel(writer, sheet_name="Total effect")
        for name, m in r["outcomes"].items():
            lr_ols_table(m).to_excel(writer, sheet_name=f"Outcome {name}"[:31])
            lr_ols_table(r["outcomes_hc3"][name]).to_excel(writer, sheet_name=f"Outcome {name} HC3"[:31])
        r["bootstrap"].to_excel(writer, sheet_name="Bootstrap NDE NIE TE", index=False)
        r["decomposition"].to_excel(writer, sheet_name="Variance decomposition")
        r["within_slopes"].to_excel(writer, sheet_name="Within-KS slopes", index=False)


def main(n_boot=2000, save_folder="/home/luisaam/Pictures"):
    summary = []
    for pr in (0.7, 0.8, 0.9):
        for drop_chance in ((False, True) if pr == 0.9 else (False,)):
            df = load_df(pr, drop_chance)
            for model in ("diracnet50", "resnet18"):
                d = df[df["model"] == model].reset_index(drop=True)
                tag = f"{model}_pr_{pr}" + ("_no_chance" if drop_chance else "")
                r = analyse(d, n_boot)
                write_excel(f"mediation_analysis_dirac_resnet18_{tag}.xlsx", d, r)
                if not drop_chance:
                    lr_mediation_plots(d, r["bootstrap"], f"{save_folder}/mediation_dirac_resnet18_{tag}")
                b = r["bootstrap"].set_index(["model", "effect"])
                dec = r["decomposition"]
                for om in LR_MEDIATION_OUTCOME_MODELS:
                    row = {"network": model, "pr": pr, "chance runs dropped": drop_chance, "n": len(d),
                           "outcome model": om, "R2": r["fit"].loc[om, "R2"]}
                    for e in ("NDE", "NIE", "TE", "NIE / TE"):
                        row[e] = b.loc[(om, e), "estimate"]
                        row[f"{e} CI"] = f"[{b.loc[(om, e), 'ci_low']:.3f}, {b.loc[(om, e), 'ci_high']:.3f}]"
                    row["dominance % saturation"] = dec.loc["scaled_saturation", "general dominance % of R2"]
                    row["dominance % kernel"] = dec.loc["Kernel_Size", "general dominance % of R2"]
                    row["mediator R2"] = r["mediator"].rsquared
                    row["mediator slope"] = r["mediator"].params["Kernel_Size"]
                    summary.append(row)
                print(f"\n===== {tag} =====")
                print(r["fit"].drop(columns="formula").round(3).to_string())
                print(r["bootstrap"].round(4).to_string())
                print(dec.round(3).to_string())
                print(r["within_slopes"].round(4).to_string())
    summary = pd.DataFrame(summary)
    summary.to_excel("mediation_analysis_dirac_resnet18_summary.xlsx", index=False)
    return summary


if __name__ == "__main__":
    s = main()
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
