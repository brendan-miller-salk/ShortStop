"""Locations used by the SAM-Consensus validation scripts. Override with environment variables."""
import os

# testingN/models/ folders from the 1000 replicate training runs
RUNS = os.environ.get("SHORTSTOP_RUNS", "MOLCELL_testing_1000_sampling")
# ShortStop features of the smORFs to score (predict mode's features/extracted_features_of_smorfs.csv)
FEATURES = os.environ.get("SHORTSTOP_FEATURES", "extracted_features_of_smorfs.csv")
# where probs.npy, orf_ids.csv, models.csv and model_auc.csv go
WORK = os.path.abspath(os.environ.get("SHORTSTOP_VALIDATION_WORK", "consensus_validation_work"))
# consensus model selection, figures and the release bundle
CONSENSUS = os.path.join(WORK, "consensus_v1.1")
