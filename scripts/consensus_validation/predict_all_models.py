"""Score the smORFs in SHORTSTOP_FEATURES with every replicate run's XGBoost model (SHORTSTOP_RUNS).

Replicates src/shortstop/prediction/predict_smorfs.py (align_features -> DNA/AA split -> scaler ->
predict_proba) for each testingN/models/ directory, reusing one precomputed feature matrix.
Writes probs.npy (n_runs x n_orfs x 3; PRISM, intracellular, secreted), orf_ids.csv, models.csv and
model_auc.csv to SHORTSTOP_VALIDATION_WORK.
"""
import os
import sys
import glob
import re
import resource
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from paths import RUNS as TESTING, FEATURES, WORK as OUT

DNA_PREFIXES = ("5_prime", "3_prime", "kozak", "first_50")
AA_DROP = ["3_prime", "5_prime", "kozak", "first_50", "label", "type", "orf_id", "local"]


def split_and_stack(df):
    dna = df.loc[:, df.columns.str.startswith(DNA_PREFIXES)].to_numpy()
    aa = df
    for pat in AA_DROP:
        aa = aa.drop(aa.filter(regex=pat).columns, axis=1)
    return np.concatenate((dna, aa.to_numpy()), axis=1)


def main():
    feats = pd.read_csv(FEATURES)
    orf_ids = feats["orf_id"]
    model_dirs = sorted(
        glob.glob(f"{TESTING}/testing*/models/best_xgb_model.model"),
        key=lambda p: int(re.search(r"testing(\d+)", p).group(1)),
    )
    print(f"{len(model_dirs)} models, {len(feats)} smORFs", flush=True)

    probs, names = [], []
    cache = {}  # aligned matrix for the most recent training column set (4 GB job limit)
    for i, mpath in enumerate(model_dirs):
        mdir = os.path.dirname(mpath)
        cols = tuple(pd.read_csv(f"{mdir}/orfs_features_in_training_model.csv", nrows=0).columns)
        if cols not in cache:
            cache.clear()
            missing = [c for c in cols if c not in feats.columns]
            aligned = feats.reindex(columns=list(cols))
            aligned[missing] = 0.0
            cache[cols] = split_and_stack(aligned)
        data = joblib.load(f"{mdir}/scaler.save").transform(cache[cols])
        model = xgb.XGBClassifier()
        model.load_model(mpath)
        probs.append(model.predict_proba(data).astype(np.float32))
        names.append(re.search(r"(testing\d+)", mdir).group(1))
        if i % 50 == 0:
            print(i, names[-1], cache[cols].shape, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "MB", flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/probs.npy", np.stack(probs))
    orf_ids.to_csv(f"{OUT}/orf_ids.csv", index=False)
    pd.Series(names, name="model").to_csv(f"{OUT}/models.csv", index=False)
    aucs = []
    for name in names:
        m = pd.read_csv(f"{TESTING}/{name}/models/macro_auc.csv")
        aucs.append(m.loc[m["model"] == "xgboost", "macro_auc"].iloc[0])
    pd.DataFrame({"model": names, "xgb_macro_auc": aucs}).to_csv(f"{OUT}/model_auc.csv", index=False)
    print("done", np.stack(probs).shape)


if __name__ == "__main__":
    sys.exit(main())
