"""
How much of DiracNet-50's identity (Dirac centre-tap) weight survives one-shot global magnitude pruning, compared with
the learned weights?

Each DiracConv2d computes with the effective kernel  K = alpha * delta + beta * normalize(W)  (see
alternate_models/diracnet.py). Before pruning, prune_models.py folds K into a plain conv weight (fold_dirac) and runs
global L1-unstructured pruning over every conv except the stem (conv1) and the classifier (linear). The identity part
I = alpha * delta lives only on the centre taps K[i, i, 1, 1], i < min(C_in, C_out); everywhere else K is purely the
learned part L = beta * normalize(W).

For every checkpoint this script splits K into I and L *before* folding, then folds and prunes with the exact same
prune_function / exclude list as prune_models.py (the mask depends only on weight magnitudes, so no data is needed),
and reports, per layer and for the whole network:

  kept_frac_all            fraction of weights kept in the pruned convs (≈ 1 - pruning rate overall)
  kept_frac_identity_taps  fraction of identity centre taps kept (an identity channel survives iff its tap is kept)
  kept_frac_learned_only   fraction of the other (purely learned) weights kept
  identity_enrichment      kept_frac_identity_taps / kept_frac_all  (> 1: identity taps are protected)
  identity_mass_kept       sum |I| over kept taps / sum |I|              (weights each tap by |alpha_i|)
  learned_energy_kept      ||L ⊙ mask||² / ||L||²                       (learned part, all positions)
  kernel_energy_kept       ||K ⊙ mask||² / ||K||²
  identity_share_dense     share of ||K||² sitting on identity taps before pruning
  identity_share_kept      same share among the kept weights after pruning
  cf_kept_frac_identity_taps   counterfactual: fraction of identity taps that would survive on their learned value
                               alone (|L_ii| above the same global threshold), i.e. without the identity boost
  threshold                global magnitude threshold (largest |K| that was pruned)

Usage on the cluster (no GPU needed; one job does all checkpoints):
  python measure_dirac_identity_pruning.py --folder "$SCRATCH/learning_rate_experiments" \
      --save_folder dirac_identity_pruning_results --pruning_rates 0.5 0.6 0.7 0.8 0.9 0.95 0.97 0.99
Restrict with --rf_levels / --lrs. Smoke test without checkpoints:  python measure_dirac_identity_pruning.py --smoke
"""
import argparse
import copy
import glob
import os
import re

import numpy as np
import omegaconf
import pandas as pd
import torch

from alternate_models.diracnet import DiracConv2d, fold_dirac
from main import prune_function
from train_CIFAR10 import get_model

LEARNING_RATES = ["0.0001", "0.001", "0.003", "0.005", "0.007", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07",
                  "0.08", "0.09", "0.1", "0.15", "0.2"]
EXCLUDE_LAYERS = ["conv1", "linear"]  # same as prune_models.main for diracnet
NAME_TEMPLATE = "lr_{lr}_recording_lr_sweep_no_ffcv"


@torch.no_grad()
def split_kernels(net):
    """{layer name: (K, I, L, identity_tap_mask)} for every DiracConv2d of an unfolded DiracNet."""
    parts = {}
    for name, m in net.named_modules():
        if isinstance(m, DiracConv2d):
            identity = m.alpha.view(-1, 1, 1, 1) * m.delta
            learned = m.beta.view(-1, 1, 1, 1) * torch.nn.functional.normalize(
                m.weight.view(m.weight.shape[0], -1)).view_as(m.weight)
            kernel = m.transform_weight()
            assert torch.allclose(kernel, identity + learned, atol=1e-6)
            parts[name] = (kernel.cpu(), identity.cpu(), learned.cpu(), (m.delta != 0).cpu())
    return parts


@torch.no_grad()
def prune_masks(folded_net, pruning_rate):
    """Prune a copy of the folded net exactly as prune_models.py does and return {layer name: kept mask} + threshold."""
    net = copy.deepcopy(folded_net)
    cfg = omegaconf.DictConfig({"amount": float(pruning_rate), "pruner": "global", "exclude_layers": EXCLUDE_LAYERS})
    prune_function(net, cfg)
    masks, pruned_mags = {}, []
    for name, m in net.named_modules():
        if hasattr(m, "weight_mask"):
            mask = m.weight_mask.bool().cpu()
            masks[name] = mask
            pruned_mags.append(m.weight_orig.detach().abs().cpu()[~mask])
    pruned_mags = torch.cat(pruned_mags)
    threshold = float(pruned_mags.max()) if pruned_mags.numel() else 0.0
    return masks, threshold


def layer_stats(K, I, L, tap, mask, threshold):
    k2, i_abs, l2 = K.pow(2), I.abs(), L.pow(2)
    kept = mask
    n_tap = int(tap.sum())
    s = {
        "n_weights": K.numel(), "n_identity_taps": n_tap,
        "n_kept": int(kept.sum()), "n_kept_identity_taps": int((kept & tap).sum()),
        "n_kept_learned_only": int((kept & ~tap).sum()), "n_learned_only": int((~tap).sum()),
        "sum_abs_identity": float(i_abs.sum()), "sum_abs_identity_kept": float((i_abs * kept).sum()),
        "learned_energy": float(l2.sum()), "learned_energy_kept": float((l2 * kept).sum()),
        "kernel_energy": float(k2.sum()), "kernel_energy_kept": float((k2 * kept).sum()),
        "kernel_energy_taps": float((k2 * tap).sum()), "kernel_energy_taps_kept": float((k2 * (tap & kept)).sum()),
        "n_cf_kept_identity_taps": int(((L.abs() > threshold) & tap).sum()),
        "mean_abs_alpha": float(I[tap].abs().mean()) if n_tap else np.nan,
        "median_abs_K_identity_taps": float(K[tap].abs().median()) if n_tap else np.nan,
        "median_abs_K_learned_only": float(K[~tap].abs().median()),
        "median_abs_L_identity_taps": float(L[tap].abs().median()) if n_tap else np.nan,
    }
    return s


def ratios(s):
    """Turn summed counts / masses into the reported fractions (works for one layer or a sum over layers)."""
    div = lambda a, b: a / b if b else np.nan
    kept_all = div(s["n_kept"], s["n_weights"])
    return {
        "kept_frac_all": kept_all,
        "kept_frac_identity_taps": div(s["n_kept_identity_taps"], s["n_identity_taps"]),
        "kept_frac_learned_only": div(s["n_kept_learned_only"], s["n_learned_only"]),
        "identity_enrichment": div(div(s["n_kept_identity_taps"], s["n_identity_taps"]), kept_all),
        "identity_mass_kept": div(s["sum_abs_identity_kept"], s["sum_abs_identity"]),
        "learned_energy_kept": div(s["learned_energy_kept"], s["learned_energy"]),
        "kernel_energy_kept": div(s["kernel_energy_kept"], s["kernel_energy"]),
        "identity_share_dense": div(s["kernel_energy_taps"], s["kernel_energy"]),
        "identity_share_kept": div(s["kernel_energy_taps_kept"], s["kernel_energy_kept"]),
        "cf_kept_frac_identity_taps": div(s["n_cf_kept_identity_taps"], s["n_identity_taps"]),
    }


SUM_KEYS = ["n_weights", "n_identity_taps", "n_kept", "n_kept_identity_taps", "n_kept_learned_only", "n_learned_only",
            "sum_abs_identity", "sum_abs_identity_kept", "learned_energy", "learned_energy_kept", "kernel_energy",
            "kernel_energy_kept", "kernel_energy_taps", "kernel_energy_taps_kept", "n_cf_kept_identity_taps"]


def analyse_net(net, meta, pruning_rates):
    parts = split_kernels(net)
    folded = fold_dirac(net)
    per_layer, summary = [], []
    for pr in pruning_rates:
        masks, threshold = prune_masks(folded, pr)
        missing = set(parts) - set(masks)
        assert not missing, f"Dirac convs not pruned: {sorted(missing)[:3]}"
        totals = dict.fromkeys(SUM_KEYS, 0)
        for depth, (name, (K, I, L, tap)) in enumerate(parts.items()):
            s = layer_stats(K, I, L, tap, masks[name], threshold)
            for k in SUM_KEYS:
                totals[k] += s[k]
            per_layer.append({**meta, "pruning_rate": float(pr), "layer": name, "layer_index": depth,
                              "threshold": threshold, **s, **ratios(s)})
        summary.append({**meta, "pruning_rate": float(pr), "threshold": threshold, "n_dirac_convs": len(parts),
                        **totals, **ratios(totals)})
    return per_layer, summary


def build_args(rf_level, dirac_width):
    return argparse.Namespace(model="diracnet50", type="normal", dataset="cifar10", RF_level=rf_level, width=1,
                              dirac_width=dirac_width)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=os.path.join(os.environ.get("SCRATCH", "."), "learning_rate_experiments"))
    p.add_argument("--save_folder", default="dirac_identity_pruning_results")
    p.add_argument("--rf_levels", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--lrs", nargs="+", default=LEARNING_RATES)
    p.add_argument("--pruning_rates", nargs="+", default=["0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.97", "0.99"])
    p.add_argument("--dirac_width", type=float, default=4)
    p.add_argument("--smoke", action="store_true", help="run on a freshly initialised DiracNet-50 (no checkpoints)")
    args = p.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)

    per_layer, summary = [], []
    if args.smoke:
        torch.manual_seed(0)
        net = get_model(build_args(1, args.dirac_width)).cpu()
        pl, sm = analyse_net(net, {"model": "diracnet50", "rf_level": 1, "lr": np.nan, "seed": 0,
                                   "checkpoint": "random init", "dense_acc": np.nan}, args.pruning_rates)
        per_layer, summary = pl, sm
        tag = "smoke"
    else:
        for rf in args.rf_levels:
            for lr in args.lrs:
                name = NAME_TEMPLATE.format(lr=lr)
                # same glob (and therefore seed order) as prune_models.py / saturation_calculation.py
                pattern = f"{args.folder}/diracnet50_normal_cifar10_*_level_{rf}_*{name}*test_acc_*.pth"
                files = glob.glob(pattern)
                if not files:
                    print(f"WARNING: no checkpoints for {pattern}")
                for seed, path in enumerate(files):
                    state = torch.load(path, map_location="cpu")
                    net = get_model(build_args(rf, args.dirac_width)).cpu()
                    net.load_state_dict(state["net"])
                    net.eval()
                    acc = re.search(r"test_acc_([0-9.]+)\.pth$", path)
                    meta = {"model": "diracnet50", "rf_level": rf, "lr": float(lr), "seed": seed,
                            "checkpoint": os.path.basename(path),
                            "dense_acc": float(acc.group(1)) if acc else state.get("acc", np.nan)}
                    pl, sm = analyse_net(net, meta, args.pruning_rates)
                    per_layer += pl
                    summary += sm
                    last = sm[-1]
                    print(f"RF{rf} lr {lr} seed {seed}: at PR {last['pruning_rate']} identity taps kept "
                          f"{last['kept_frac_identity_taps']:.3f} vs learned-only {last['kept_frac_learned_only']:.4f} "
                          f"(counterfactual {last['cf_kept_frac_identity_taps']:.3f})", flush=True)
        tag = "rf_" + "_".join(map(str, args.rf_levels)) + ("" if args.lrs == LEARNING_RATES else "_lr_" + "_".join(args.lrs))

    per_layer, summary = pd.DataFrame(per_layer), pd.DataFrame(summary)
    per_layer.to_csv(os.path.join(args.save_folder, f"diracnet50_identity_pruning_per_layer_{tag}.csv"), index=False)
    summary.to_csv(os.path.join(args.save_folder, f"diracnet50_identity_pruning_summary_{tag}.csv"), index=False)
    cols = ["kept_frac_all", "kept_frac_identity_taps", "kept_frac_learned_only", "identity_enrichment",
            "identity_mass_kept", "learned_energy_kept", "identity_share_dense", "identity_share_kept",
            "cf_kept_frac_identity_taps"]
    if len(summary):
        print("\nMean over checkpoints, by pruning rate:")
        print(summary.groupby("pruning_rate")[cols].mean().round(4).to_string())
        print(f"\nWrote {len(summary)} summary rows and {len(per_layer)} per-layer rows to {args.save_folder}/")


if __name__ == "__main__":
    main()
