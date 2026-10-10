"""Select the 100 ShortStop consensus models and write consensus_manifest.csv.

model_001 is the published standard model (testing251). testing144 is excluded by the author.
model_002..model_100 are the next 99 runs by XGBoost macro AUC.
Per-model stats are on gencode_smorfs with a SAM vote = P(intracellular)+P(secreted) > 0.5.
"""
import os
import numpy as np
import pandas as pd

from paths import WORK as AGREE, CONSENSUS as HERE
os.makedirs(HERE, exist_ok=True)
STANDARD = "testing251"
EXCLUDED = {"testing144": "excluded by author (anomaly found on inspection)"}
N_MODELS = 100
SAM_PROB = 0.5

auc = pd.read_csv(f"{AGREE}/model_auc.csv").sort_values("xgb_macro_auc", ascending=False)
pool = auc[~auc["model"].isin(EXCLUDED)]
others = pool[pool["model"] != STANDARD].head(N_MODELS - 1)
chosen = pd.concat([pool[pool["model"] == STANDARD], others], ignore_index=True)

probs = np.load(f"{AGREE}/probs.npy")
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
