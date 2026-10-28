"""Select the 100 ShortStop consensus models and write consensus_manifest.csv.

model_001 is the published standard model, linked into SHORTSTOP_RUNS as testing0 (SHORTSTOP_STANDARD).
model_002..model_100 are the next 99 runs by XGBoost macro AUC. Runs listed in SHORTSTOP_EXCLUDED
("run:reason;run:reason") are left out; none are for v1.2.
Per-model stats are on GENCODE smORF IDs (starting with lowercase "c") with a SAM vote =
P(intracellular)+P(secreted) > 0.5.
"""
import os
import numpy as np
import pandas as pd

from paths import WORK as AGREE, CONSENSUS as HERE
os.makedirs(HERE, exist_ok=True)
STANDARD = os.environ.get("SHORTSTOP_STANDARD", "testing0")
EXCLUDED = dict(x.split(":", 1) for x in os.environ.get("SHORTSTOP_EXCLUDED", "").split(";") if x)
N_MODELS = 100
SAM_PROB = 0.5

auc = pd.read_csv(f"{AGREE}/model_auc.csv").sort_values("xgb_macro_auc", ascending=False)
pool = auc[~auc["model"].isin(EXCLUDED)]
others = pool[pool["model"] != STANDARD].head(N_MODELS - 1)
chosen = pd.concat([pool[pool["model"] == STANDARD], others], ignore_index=True)

gencode = pd.read_csv(f"{AGREE}/orf_ids.csv")["orf_id"].str.strip('"').str.match(r"^c").to_numpy()
probs = np.load(f"{AGREE}/probs.npy")[:, gencode]
runs = pd.read_csv(f"{AGREE}/models.csv")["model"]
idx = [runs[runs == m].index[0] for m in chosen["model"]]
sam = probs[idx, :, 1:].sum(2) > SAM_PROB
X = sam.astype(np.float32)
inter = X @ X.T
size = X.sum(1)
jac = inter / (size[:, None] + size[None, :] - inter)

manifest = pd.DataFrame({
    "model_id": [f"model_{i:03d}" for i in range(1, N_MODELS + 1)],
    "source_run": chosen["model"],
    "xgb_macro_auc": chosen["xgb_macro_auc"].round(4),
    "is_standard": chosen["model"] == STANDARD,
    "n_sam_calls_gencode": size.astype(int),
    "mean_jaccard_to_others": ((jac.sum(1) - 1) / (N_MODELS - 1)).round(3),
    "notes": np.where(chosen["model"] == STANDARD, "published standard prediction model", ""),
})
manifest.to_csv(f"{HERE}/consensus_manifest.csv", index=False)
excluded = auc[auc["model"].isin(EXCLUDED)].assign(reason=lambda d: d["model"].map(EXCLUDED))
excluded.to_csv(f"{HERE}/excluded_models.csv", index=False)
pd.DataFrame({"model": chosen["model"]}).to_csv(f"{HERE}/models.csv", index=False)

print(manifest.head(5).to_string(index=False))
print(f"... AUC range {manifest.xgb_macro_auc.min()}-{manifest.xgb_macro_auc.max()}")
