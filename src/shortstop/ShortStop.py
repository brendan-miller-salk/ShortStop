import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib
import random
import pathlib
import urllib.request

from shortstop.pipeline import Pipeline

# Locate where the package was installed
BASE_DIR = pathlib.Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR / 'demo_data'
MODEL_DIR = BASE_DIR / 'standard_prediction_model'

# Auto-download if missing
def download_demo_data():
    DEMO_DIR.mkdir(exist_ok=True)
    gtf_path = DEMO_DIR / 'gencode.v43.primary_assembly.basic.annotation.gtf'
    fa_path = DEMO_DIR / 'hg_38_primary.fa'

    if not gtf_path.exists():
        print("⬇️ Downloading GTF...")
        urllib.request.urlretrieve(
            "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.primary_assembly.basic.annotation.gtf.gz",
            DEMO_DIR / "temp.gtf.gz"
        )
        os.system(f"gunzip -f {DEMO_DIR}/temp.gtf.gz")
        os.system(f"grep -v '^#' {DEMO_DIR}/temp.gtf > {gtf_path}")
        os.remove(DEMO_DIR / 'temp.gtf')

    if not fa_path.exists():
        print("⬇️ Downloading genome FASTA...")
        urllib.request.urlretrieve(
            "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/GRCh38.primary_assembly.genome.fa.gz",
            DEMO_DIR / "temp.fa.gz"
        )
        os.system(f"gunzip -f {DEMO_DIR}/temp.fa.gz")
        os.rename(DEMO_DIR / 'temp.fa', fa_path)

class ShortStop:
    def __init__(self):
        self.args = self.__get_args()

    def __get_args(self):
        self.main_parser = argparse.ArgumentParser(
            description="ShortStop",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self.mode_parser = self.main_parser.add_argument_group("Mode input options")
        self.mode_parser.add_argument("mode", metavar="Mode", help=(
            "Mode to run the pipeline for.\nList of Modes: "
            "train, generate_insilico_decoy_sequences, feature_extract, predict, train_with_custom_features, demo"
        ))

        # Parse first positional arg to determine mode
        initial_args = self.main_parser.parse_args(sys.argv[1:2])
        self.mode = initial_args.mode

        # Now construct parser for full mode
        self.parser = argparse.ArgumentParser(
            description=f"Run pipeline in {self.mode} mode",
            prog=" ".join(sys.argv[0:2]),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self.general_args = self.parser.add_argument_group("General Parameters")
        self.general_args.add_argument("mode", metavar=self.mode)
        self.general_args.add_argument("--outdir", "-o", help="Inform the output directory", default="shortstop_output")
        self.general_args.add_argument("--threads", "-p", help="Number of threads to be used.", default=1)

        # Define mode-specific args
        self.__configure_mode()

        # NOW parse everything (after mode args are defined)
        self.args = self.parser.parse_args()

        # Conditionally download demo data (after args are available)
        if self.mode == 'demo':
            if not self.args.positive_gtf or not pathlib.Path(self.args.positive_gtf).exists() \
            or not self.args.genome or not pathlib.Path(self.args.genome).exists():
                download_demo_data()

        return self.args

    def __configure_mode(self):
        if self.mode == 'train' or self.mode == 'insilico' or self.mode == 'feature_extract':
            self.__set_train_mode()
        elif self.mode == 'predict':
            self.__set_predict_mode()
        elif self.mode == 'demo':
            self.__set_demo_mode()
            
    def __set_train_mode(self):
        self.modeArguments = self.parser.add_argument_group("Training mode options")
        self.modeArguments.add_argument("--positive_gtf", help="Reference gtf file", default=str(DEMO_DIR / 'gencode.v43.primary_assembly.basic.annotation.gtf'))
        self.modeArguments.add_argument("--positive_ids", help="Positive ORF ID file", default=str(DEMO_DIR / 'uniprot_entry_ensembl_ids.csv'))
        self.modeArguments.add_argument("--positive_functions", help="Protein function CSV", default=str(DEMO_DIR / 'shortstop_train_cc.csv'))
        self.modeArguments.add_argument("--genome", help="Genome fasta file", default=str(DEMO_DIR / 'hg_38_primary.fa'))
        self.modeArguments.add_argument("--putative_smorfs_gtf", help="smORF GTF", default=str(DEMO_DIR / 'gencode_smorfs.gtf'))
        self.modeArguments.add_argument("--utr_length", default=25)
        self.modeArguments.add_argument("--n_insilico_smORFs", default=1000)
        self.modeArguments.add_argument("--kmer", default=4)

    def __set_predict_mode(self):
        self.modeArguments = self.parser.add_argument_group("Predict mode options")
        self.modeArguments.add_argument("--genome", default=str(DEMO_DIR / 'hg_38_primary.fa'))
        self.modeArguments.add_argument("--putative_smorfs_gtf", default=str(DEMO_DIR / 'chr1_smorfs.gtf'))
        self.modeArguments.add_argument("--utr_length", default=25)
        self.modeArguments.add_argument("--kmer", default=4)
        self.modeArguments.add_argument("--orfs_features_in_train_model", default=str(MODEL_DIR / 'orfs_features_in_train_model.csv'))
        self.modeArguments.add_argument("--orfs_to_be_predicted", default='shortstop_output/features/extracted_features_of_smorfs.csv')
        self.modeArguments.add_argument("--model_scaler", default=str(MODEL_DIR / 'scaler.save'))
        self.modeArguments.add_argument("--model", default=str(MODEL_DIR / 'best_xgb_model.model'))

    def __set_demo_mode(self):
        self.modeArguments = self.parser.add_argument_group("Demo mode options")
        self.modeArguments.add_argument("--positive_gtf", default=str(DEMO_DIR / 'gencode.v43.primary_assembly.basic.annotation.gtf'))
        self.modeArguments.add_argument("--positive_ids", default=str(DEMO_DIR / 'uniprot_entry_ensembl_ids.csv'))
        self.modeArguments.add_argument("--positive_functions", default=str(DEMO_DIR / 'shortstop_train_cc.csv'))
        self.modeArguments.add_argument("--genome", default=str(DEMO_DIR / 'hg_38_primary.fa'))
        self.modeArguments.add_argument("--putative_smorfs_gtf", default=str(DEMO_DIR / 'gencode_smorfs.gtf'))
        self.modeArguments.add_argument("--utr_length", default=25)
        self.modeArguments.add_argument("--n_insilico_smORFs", default=200)
        self.modeArguments.add_argument("--kmer", default=2)
        self.modeArguments.add_argument("--orfs_features_in_train_model", default=str(MODEL_DIR / 'orfs_features_in_train_model.csv'))
        self.modeArguments.add_argument("--orfs_to_be_predicted", default='shortstop_output/features/extracted_features_of_smorfs.csv')
        self.modeArguments.add_argument("--model_scaler", default=str(MODEL_DIR / 'scaler.save'))
        self.modeArguments.add_argument("--model", default=str(MODEL_DIR / 'best_xgb_model.model'))

    def execute(self):
        if self.mode in ['train', 'insilico', 'feature_extract']:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            pipeline = Pipeline(args=self.args)
            pipeline.train()
        elif self.mode == 'predict':
            pipeline = Pipeline(args=self.args)
            pipeline.predict()
        elif self.mode == 'demo':
            pipeline = Pipeline(args=self.args)
            pipeline.demo()
            


if __name__ == '__main__':
    print("""     _______. __    __    ______   .______     .___________.    _______.___________.  ______   .______   
    /       ||  |  |  |  /  __  \  |   _  \    |           |   /       |           | /  __  \  |   _  \  
   |   (----`|  |__|  | |  |  |  | |  |_)  |   `---|  |----`  |   (----`---|  |----`|  |  |  | |  |_)  | 
    \   \    |   __   | |  |  |  | |      /        |  |        \   \       |  |     |  |  |  | |   ___/  
.----)   |   |  |  |  | |  `--'  | |  |\  \----.   |  |    .----)   |      |  |     |  `--'  | |  |      
|_______/    |__|  |__|  \______/  | _| `._____|   |__|    |_______/       |__|      \______/  | _|      
           """)

    print("""ShortStop classifies translating smORFs as 'Swiss-Prot Analog Microproteins' (SAMs) or "Physicochemically Resembling In Silico Microproteins' (PRISMs). Users can train their own model, create biochemically-matched null sequences through in silico microprotein generation, and extract features for use outside ShortStop.""")
    shortstop = ShortStop()
    shortstop.execute()