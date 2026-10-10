"""Package the 100 consensus models as shortstop_consensus_v1.1.tar.gz (deterministic) + sha256.

Each model_XXX/ holds best_xgb_model.model, scaler.save, and orfs_features_in_train_model.csv
(header line only, which is all predict mode reads). consensus_manifest.csv sits at the top level.
"""
import gzip
import hashlib
import io
import os
import shutil
import tarfile
import pandas as pd

from paths import RUNS, CONSENSUS as HERE
NAME = "shortstop_consensus_v1.1"
STAGE = f"{HERE}/{NAME}"

manifest = pd.read_csv(f"{HERE}/consensus_manifest.csv")
if os.path.isdir(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)
for model_id, run in zip(manifest["model_id"], manifest["source_run"]):
    src, dst = f"{RUNS}/{run}/models", f"{STAGE}/{model_id}"
    os.makedirs(dst)
    shutil.copyfile(f"{src}/best_xgb_model.model", f"{dst}/best_xgb_model.model")
    shutil.copyfile(f"{src}/scaler.save", f"{dst}/scaler.save")
    with open(f"{src}/orfs_features_in_training_model.csv") as fin, \
         open(f"{dst}/orfs_features_in_train_model.csv", "w") as fout:
        fout.write(fin.readline())
shutil.copyfile(f"{HERE}/consensus_manifest.csv", f"{STAGE}/consensus_manifest.csv")


def add(tar, path, arcname):
    info = tar.gettarinfo(path, arcname)
    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
    info.mode = 0o755 if info.isdir() else 0o644
    if info.isdir():
        tar.addfile(info)
    else:
        with open(path, "rb") as f:
            tar.addfile(info, f)


buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
    add(tar, STAGE, NAME)
    for root, dirs, files in os.walk(STAGE):
        dirs.sort()
        for d in dirs:
            add(tar, f"{root}/{d}", os.path.relpath(f"{root}/{d}", HERE))
        for fn in sorted(files):
            add(tar, f"{root}/{fn}", os.path.relpath(f"{root}/{fn}", HERE))
out = f"{HERE}/{NAME}.tar.gz"
with open(out, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
    gz.write(buf.getvalue())
sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
open(f"{out}.sha256", "w").write(f"{sha}  {NAME}.tar.gz\n")
print(f"{out}  {os.path.getsize(out) / 1e6:.1f} MB\nsha256 {sha}")
