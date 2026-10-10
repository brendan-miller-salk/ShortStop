"""Figures and validation tables for the ShortStop v1.1 consensus models (light and dark figure variants).

1. consensus_auc_{theme}.png     - XGBoost macro AUC of the replicate training runs, consensus models highlighted
2. consensus_elbow_{theme}.png   - SAM-Consensus list overlap between two independent model sets vs size
3. consensus_levels_{theme}.png  - SAM-Consensus percentage per smORF, with the 50 / 90 / 100 levels
agreement_stats.csv, elbow.csv and random_comparison.csv hold the numbers behind docs/consensus_validation.md.
Each model votes SAM when P(intracellular) + P(secreted) > 0.5. SAM-Consensus-50: > 50% of models,
-90: >= 90%, -100: all models.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import WORK as AGREE, CONSENSUS as HERE
OUT = f"{HERE}/figures"
SAM_PROB = 0.5

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", rest="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834", ramp=("#86b6ef", "#2a78d6", "#104281")),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", rest="#52514e",
                 s1="#3987e5", s2="#d95926", ramp=("#184f95", "#3987e5", "#9ec5f4")),
}

manifest = pd.read_csv(f"{HERE}/consensus_manifest.csv")
excluded = pd.read_csv(f"{HERE}/excluded_models.csv")
auc = pd.read_csv(f"{AGREE}/model_auc.csv")
probs = np.load(f"{AGREE}/probs.npy")
runs = pd.read_csv(f"{AGREE}/models.csv")["model"]
vote_all = probs[:, :, 1:].sum(2) > SAM_PROB  # runs x smORFs
pool_idx = np.flatnonzero(~runs.isin(excluded["model"]).to_numpy())
ens_idx = np.array([runs[runs == m].index[0] for m in manifest["source_run"]])
n_votes = vote_all[ens_idx].sum(0)  # integer SAM votes per smORF (0-100)
pct = 100 * n_votes / len(ens_idx)


def levels(votes):
    """SAM-Consensus-50 / -90 / -100 membership from a (models x smORFs) vote matrix."""
    k, n = votes.sum(0), votes.shape[0]
    return {50: 2 * k > n, 90: 10 * k >= 9 * n, 100: k == n}  # integer arithmetic: > 50%, >= 90%, all


def mean_jaccard(votes):
    X = votes.astype(np.float32)
    inter = X @ X.T
    s = X.sum(1)
    return (inter / (s[:, None] + s[None, :] - inter))[np.triu_indices(len(X), 1)].mean()


def elbow(sizes, reps=50, seed=0):
    """Jaccard between SAM-Consensus lists of two disjoint random model sets of size n."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        res = {50: [], 90: []}
        for _ in range(reps):
            pick = rng.permutation(pool_idx)[: 2 * n]
            la, lb = levels(vote_all[pick[:n]]), levels(vote_all[pick[n:]])
            for k in res:
                res[k].append((la[k] & lb[k]).sum() / max((la[k] | lb[k]).sum(), 1))
        row = {"n_models": n}
        for k in res:
            p5, med, p95 = np.percentile(res[k], [5, 50, 95])
            row.update({f"sam_consensus_{k}_p5": p5, f"sam_consensus_{k}_median": med, f"sam_consensus_{k}_p95": p95})
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def random_comparison(reps=200, seed=0):
    """The shipped consensus models vs random 100-model sets drawn from all completed runs."""
    rng = np.random.default_rng(seed)

    def stats(v):
        lv = levels(v)
        return [mean_jaccard(v), lv[50].sum(), lv[90].sum(), lv[100].sum()]

    R = np.array([stats(vote_all[rng.choice(pool_idx, 100, replace=False)]) for _ in range(reps)])
    names = ["mean_pairwise_jaccard", "sam_consensus_50", "sam_consensus_90", "sam_consensus_100"]
    return pd.DataFrame({"metric": names, "shipped_consensus": stats(vote_all[ens_idx]),
                         "random_median": np.median(R, 0), "random_p2.5": np.percentile(R, 2.5, 0),
                         "random_p97.5": np.percentile(R, 97.5, 0)}).round(3)


def agreement_stats():
    """Agreement among the shipped consensus models (same vote rule as ShortStop's consensus_table)."""
    v = vote_all[ens_idx]
    n, m = v.shape
    X = v.astype(np.float32)
    inter = X @ X.T
    size = X.sum(1)
    iu = np.triu_indices(n, 1)
    jac = (inter / (size[:, None] + size[None, :] - inter))[iu]
    k = v.sum(0)
    pairs = (k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))
    p_sam = v.mean()
    p_chance = p_sam ** 2 + (1 - p_sam) ** 2
    ranks = pd.DataFrame(probs[ens_idx, :, 1:].sum(2)).rank(axis=1).to_numpy()
    rows = {
        "n_models": n, "n_smorfs": m,
        "sam_votes_per_model_median": np.median(size), "sam_votes_per_model_min": size.min(),
        "sam_votes_per_model_max": size.max(), "sam_vote_fraction": p_sam,
        "mean_pairwise_jaccard": jac.mean(), "pairwise_jaccard_p5": np.percentile(jac, 5),
        "pairwise_jaccard_p95": np.percentile(jac, 95), "dice_from_mean_jaccard": 2 * jac.mean() / (1 + jac.mean()),
        "mean_pairwise_containment": (inter / size[:, None])[~np.eye(n, dtype=bool)].mean(),
        "fleiss_kappa": (pairs.mean() - p_chance) / (1 - p_chance),
        "mean_pairwise_spearman_p_sam": np.corrcoef(ranks)[iu].mean(),
    }
    return pd.DataFrame({"metric": list(rows), "value": list(rows.values())}).round(4)


SIZES = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 460]
agreement_stats().to_csv(f"{HERE}/agreement_stats.csv", index=False)
EL = elbow(SIZES)
EL.to_csv(f"{HERE}/elbow.csv", index=False)
random_comparison().to_csv(f"{HERE}/random_comparison.csv", index=False)


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.yaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def title(ax, t, main, sub):
    ax.set_title(main, loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=t["ink2"], fontsize=9, va="bottom")


def fig_auc(t):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=t["surface"])
    style(ax, t)
    bins = np.arange(0.950, 0.9875, 0.001)
    sel = auc["model"].isin(manifest["source_run"])
    ax.hist([auc.loc[sel, "xgb_macro_auc"], auc.loc[~sel, "xgb_macro_auc"]], bins=bins,
            stacked=True, color=[t["s1"], t["rest"]], edgecolor=t["surface"], linewidth=1,
            label=["Consensus models (100)", "Other replicate runs"])
    std = manifest.loc[manifest.is_standard, "xgb_macro_auc"].iloc[0]
    ymax = ax.get_ylim()[1]
    ax.axvline(std, color=t["s2"], lw=2, ymax=0.62)
    ax.annotate("Standard model (model_001)", xy=(std, ymax * 0.62), xytext=(std - 0.0105, ymax * 0.80),
                color=t["ink2"], fontsize=8.5, arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8))
    ax.set_xlabel("XGBoost macro AUC (each run's own held-out test split)")
    ax.set_ylabel("Training runs")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=t["ink2"])
    title(ax, t, "Model selection: AUC of replicate ShortStop training runs",
          "Shipped: the standard model plus the 99 next-highest-AUC replicates.")
    fig.tight_layout()
    return fig


def fig_elbow(t):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=t["surface"])
    style(ax, t)
    for k, c in ((50, t["s2"]), (90, t["s1"])):
        lo, med, hi = EL[f"sam_consensus_{k}_p5"], EL[f"sam_consensus_{k}_median"], EL[f"sam_consensus_{k}_p95"]
        ax.fill_between(SIZES, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(SIZES, med, color=c, lw=2, marker="o", ms=4.5,
                markeredgecolor=t["surface"], markeredgewidth=1.5, label=f"SAM-Consensus-{k}")
        m100 = float(med[EL.n_models == 100].iloc[0])
        ax.annotate(f"{m100:.2f}", (100, m100), textcoords="offset points",
                    xytext=(6, 6 if k == 50 else -12), color=t["ink2"], fontsize=8.5)
    ax.axvline(100, color=t["muted"], lw=1, ls="--")
    ax.text(104, 0.52, "shipped consensus (100)", color=t["ink2"], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200, 460])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "460"])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Number of models")
    ax.set_ylabel("List overlap between two\nindependent model sets (Jaccard)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=t["ink2"])
    title(ax, t, "Elbow: how reproducible are the SAM-Consensus lists?",
          "Two disjoint random model sets per size, 50 draws (median, 5–95% band). Gains flatten past ~100.")
    fig.tight_layout()
    return fig


def fig_levels(t):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=t["surface"])
    style(ax, t)
    # Integer vote percentages binned so no bar straddles a level: 1-5 ... 46-50 | 51-55 ... 86-89 | 90-94, 95-99 | 100
    groups = ([(a + 1, a + 5) for a in range(0, 50, 5)] + [(a + 1, min(a + 5, 89)) for a in range(50, 90, 5)]
              + [(90, 94), (95, 99), (100, 100)])
    assert len(ens_idx) == 100, "bins assume 100 models (one vote = one percentage point)"
    counts = [((n_votes >= g0) & (n_votes <= g1)).sum() for g0, g1 in groups]
    assert sum(counts) == (n_votes > 0).sum()
    lefts = list(range(0, 105, 5))
    lo, mid, hi = t["ramp"]
    colors = [hi if g0 == 100 else mid if g0 >= 90 else lo if g0 > 50 else t["rest"] for g0, _ in groups]
    ax.bar(lefts, counts, width=5, align="edge", color=colors, edgecolor=t["surface"], linewidth=2)
    lv = {k: v.sum() for k, v in levels(vote_all[ens_idx]).items()}
    ymax = max(counts)
    for x, lab in ((25, f"Not SAM-Consensus-50 (1–50%)\n{((n_votes > 0) & (n_votes <= 50)).sum()} smORFs"),
                   (70, f"SAM-Consensus-50\n{lv[50]} (> 50%)"), (95, f"-90\n{lv[90]}"), (102.5, f"-100\n{lv[100]}")):
        ax.text(x, ymax * 1.03, lab, ha="center", va="bottom", color=t["ink2"], fontsize=8.5)
    for x in (50, 90, 100):
        ax.axvline(x, color=t["muted"], lw=1, ls="--")
    ax.set_ylim(0, ymax * 1.28)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 20, 40, 50, 60, 80, 90, 102.5])
    ax.set_xticklabels(["0", "20", "40", "50", "60", "80", "90", "100"])
    ax.set_xlabel("SAM-Consensus percentage (% of the 100 models with P(SAM) > 0.5)")
    ax.set_ylabel("smORFs")
    title(ax, t, "SAM-Consensus levels on 7,155 GENCODE smORFs",
          f"Nested: every -100 smORF is also -90 and -50. {(pct == 0).sum():,} smORFs with no SAM votes not shown.")
    fig.tight_layout()
    return fig


os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))
for name, fn in (("consensus_auc", fig_auc), ("consensus_elbow", fig_elbow), ("consensus_levels", fig_levels)):
    for theme, t in THEMES.items():
        fig = fn(t)
        fig.savefig(f"{OUT}/{name}_{theme}.png", dpi=200, facecolor=t["surface"])
        plt.close(fig)
print(pd.read_csv(f"{HERE}/agreement_stats.csv").to_string(index=False))
print(EL.to_string(index=False))
print(pd.read_csv(f"{HERE}/random_comparison.csv").to_string(index=False))
