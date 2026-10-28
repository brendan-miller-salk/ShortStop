# ShortStop SAM-Consensus validation

How the 100 SAM-Consensus models shipped with ShortStop v1.2 were chosen, and how far they agree.
All numbers are for the 7,152 GENCODE smORFs that ShortStop scored from
`src/shortstop/demo_data/gencode_smorfs.gtf`.

## The 100 models

ShortStop was trained 1,000 times with `shortstop train --n_insilico_smORFs 1000` and the v1.2
sequence extraction. Each run drew its own random PRISM sample and train/test split. Runs were
ranked by the XGBoost macro AUC each one measured on its own held-out test split. The 100 consensus
models are:

- `model_001`: the published standard model (unchanged);
- `model_002`–`model_100`: the 99 highest-AUC runs.

Their AUCs range from 0.979 to 0.987. `consensus_manifest.csv`, shipped inside the model bundle,
lists every model with its source run, AUC, SAM votes on GENCODE smORFs and mean overlap with the
other 99. The in silico PRISMs each model was trained against are attached to the v1.2.0 release
(`shortstop_consensus_v1.2_insilico_prisms.tar.gz`).

Each run's AUC comes from a different random test split, so differences of a few thousandths are
partly split-to-split noise. Selecting by AUC helps only a little (see below).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/consensus_auc_dark.png">
  <img alt="AUC distribution of the replicate training runs" src="figures/consensus_auc_light.png" width="700">
</picture>

## SAM and SAM-Consensus labels

Each consensus model votes SAM when P(intracellular) + P(secreted) > 0.5. The SAM-Consensus
percentage is the share of the 100 models voting SAM.

| Label | Rule | smORFs | Intracellular / secreted | Also SAM (standard) |
|---|---|---|---|---|
| SAM | standard model classification | 604 | 570 / 34 | – |
| SAM-Consensus-50 | > 50% of models | 522 | 493 / 29 | 438 |
| SAM-Consensus-90 | ≥ 90% of models | 282 | 262 / 20 | 269 |
| SAM-Consensus-100 | all 100 models | 126 | 114 / 12 | 126 |

For the SAM-Consensus labels, intracellular vs secreted is the majority among the SAM-voting
models; ties go to intracellular (`class_agreement` = 0.5).

**Why model_001 votes SAM for 630 smORFs but the standard model calls 604 SAMs.** The two use
different rules. The standard classification keeps v1.0's rule: the single most probable class.
A consensus vote counts P(intracellular) + P(secreted) > 0.5. For 26 smORFs, P(SAM) is above 0.5,
but PRISM is still the single most probable class (for example PRISM 0.45, intracellular 0.30,
secreted 0.25). The reverse never happens.

**How the standard SAMs spread across the levels.** Of the 604 standard-model SAMs:
- 126 are SAM-Consensus-100;
- 143 more are -90;
- 169 more are -50;
- 166 are not SAM-Consensus-50. These depend on the particular training sample.

**The reverse case.** 84 smORFs are SAM-Consensus-50 although the standard model classifies them as
PRISM.

**Localization.** 23 SAM-Consensus-90 smORFs (9 of them SAM-Consensus-100) have under 90% of their
SAM-voting models agreeing on intracellular vs secreted (`class_agreement` < 0.9). Treat the
localization label of those as uncertain.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/consensus_levels_dark.png">
  <img alt="SAM-Consensus percentage per smORF with the 50, 90 and 100 levels marked" src="figures/consensus_levels_light.png" width="700">
</picture>

## Agreement between models

| Each model votes SAM when P(SAM) > 0.5 | |
|---|---|
| SAM votes per model, median (range) | 570 (439–832) |
| Mean pairwise Jaccard of SAM lists (5–95%) | 0.576 (0.524–0.622) |
| Fleiss' κ (SAM vs not) | 0.707 |
| Mean pairwise Spearman of P(SAM) | 0.869 |

**How to read these.**
- **Jaccard** is the size of the overlap between two models' SAM lists divided by the size of their
  combination. At 0.576, about 73% of one model's SAM votes are also made by another model. The
  rest depend on the particular training sample.
- **Fleiss' κ** is agreement over all smORFs, corrected for chance. 0.61–0.80 is conventionally
  "substantial".
- **The two are not independent evidence.** SAMs are rare (~8% of smORFs here), so the two-category
  κ is close to the Dice overlap 2J / (1 + J) = 0.731.
- **The models rank smORFs consistently** (Spearman 0.87). They disagree mainly about smORFs near
  P(SAM) = 0.5. The SAM-Consensus-90 and -100 labels remove that sensitivity by requiring almost
  every model to agree.

## Does selecting by AUC matter?

The shipped consensus models compared with 200 random sets of 100 replicate runs:

| | Shipped consensus models | Random 100, median [95% range] |
|---|---|---|
| Mean pairwise Jaccard | 0.576 | 0.562 [0.556–0.568] |
| SAM-Consensus-50 | 522 | 501 [488–512] |
| SAM-Consensus-90 | 282 | 260 [249–270] |
| SAM-Consensus-100 | 126 | 117 [109–124] |

Higher-AUC models agree slightly more, which yields somewhat longer SAM-Consensus-90 lists. But
the labels come mostly from counting how many models agree, not from choosing particular models.

## Why 100 models (the elbow)

Two sets of models were drawn from disjoint random sets of replicate runs, and the overlap (Jaccard)
between their SAM-Consensus lists was measured. Values are the median and 5–95% range over 50
draws.

| Models per set | SAM-Consensus-50 | SAM-Consensus-90 |
|---|---|---|
| 5 | 0.73 (0.70–0.76) | 0.66 (0.62–0.68) |
| 10 | 0.79 (0.76–0.81) | 0.75 (0.71–0.78) |
| 30 | 0.88 (0.86–0.90) | 0.82 (0.78–0.85) |
| 50 | 0.90 (0.89–0.92) | 0.85 (0.83–0.88) |
| **100** | **0.93 (0.92–0.95)** | **0.89 (0.86–0.92)** |
| 200 | 0.95 (0.94–0.96) | 0.92 (0.90–0.94) |
| 460 | 0.97 (0.96–0.98) | 0.95 (0.94–0.97) |

Gains flatten past about 100 models. Doubling to 200 improves the overlap by 0.02–0.03, at twice
the download size and run time.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/consensus_elbow_dark.png">
  <img alt="SAM-Consensus list reproducibility versus the number of models" src="figures/consensus_elbow_light.png" width="700">
</picture>

## Output: `predictions/consensus/shortstop_consensus.csv`

| Column | Meaning |
|---|---|
| `orf_id` | smORF identifier |
| `classification`, `sam_probability` | standard model call and P(SAM), as in `shortstop_classifications.csv` / `sams.csv` |
| `sam` | standard model classifies the smORF as a SAM (`sam_intracellular` or `sam_secreted`) |
| `sam_consensus_percentage` | % of the 100 models with P(SAM) > 0.5 |
| `sam_consensus_50`, `sam_consensus_90`, `sam_consensus_100` | > 50%, ≥ 90% and 100% of models vote SAM |
| `n_models_sam` | number of models voting SAM |
| `mean_sam_probability`, `sd_sam_probability` | P(SAM) across the 100 models |
| `consensus_class` | `sam_intracellular` or `sam_secreted`: the majority among SAM-voting models; ties go to intracellular (blank if no votes) |
| `class_agreement` | fraction of SAM-voting models that agree with `consensus_class` |

`consensus_summary.txt` reports the label counts, per-model SAM votes, pairwise Jaccard and Fleiss'
κ for your own input. It also checks that `model_001` reproduces the standard model.

## Caveats

- **Agreement measures stability, not correctness.** All replicates share the same positive training
  set (Swiss-Prot microproteins) and the same features. Any bias those carry is shared by every
  model.
- **The SAM-Consensus levels were characterized on human GENCODE smORFs**, the training species.
  They carry over to other inputs only as far as the standard model does.

## Reproducing

The scripts are in [`scripts/consensus_validation/`](../scripts/consensus_validation/). They need the
replicate training-run folders, which are not distributed. Given those, the scripts rebuild the
bundle byte-for-byte (sha256 `4f77162b…4298e835`).
