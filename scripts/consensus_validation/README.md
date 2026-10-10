# SAM-Consensus validation scripts

These scripts rebuild the ShortStop v1.1 consensus model bundle and the numbers in
[docs/consensus_validation.md](../../docs/consensus_validation.md). Run them in the ShortStop
environment, in this order:

```bash
export SHORTSTOP_RUNS=/path/to/MOLCELL_testing_1000_sampling       # testingN/models/ per training run
export SHORTSTOP_FEATURES=/path/to/extracted_features_of_smorfs.csv  # features of the smORFs to score
export SHORTSTOP_VALIDATION_WORK=/path/to/work                       # outputs

python predict_all_models.py           # score every run's XGBoost model -> probs.npy, model_auc.csv
python select_models.py                # pick the 100 -> consensus_v1.1/consensus_manifest.csv
python consensus_figures.py            # figures, agreement_stats.csv, elbow.csv, random_comparison.csv
python build_bundle.py                 # shortstop_consensus_v1.1.tar.gz + .sha256
```

`predict_all_models.py` uses about 700 MB of memory and takes about 15 minutes for ~1,000 models on
7,155 smORFs. The other scripts take a few minutes in total.
