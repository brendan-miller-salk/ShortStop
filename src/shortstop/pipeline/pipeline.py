import os
import shutil
from ..training import SequenceExtractor, NegativeSet, DatabaseCombiner, FeatureExtractor, UMAPVisualizer
from ..prediction import smORFPredictor

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.outdir = args.outdir
    
    def train(self):
        if self.args.mode == 'train':
            print("▶️ You have initiated training...")
        elif self.args.mode == 'insilico':
            print("▶️ You have initiated insilico sequence generation...")
        elif self.args.mode == 'feature_extract':
            print("▶️ You have initiated feature extraction...")
            
        seq_extractor = SequenceExtractor(args=self.args)
        seq_extractor.create_positive_gtf()
        print("✅ GTF for positive ORFs completed.")
        seq_extractor.extract_sequences()
        print("✅ Amino acids and DNA for each ORF extracted.")

        print("⏳Generating pseudo-insilico sequences based on your putative smORFs...")
        decoy = NegativeSet(args=self.args)
        decoy.turn_two()
        decoy.combine_databases()
        print("✅ Pseudo-insilico decoy sequences generated.")
        
        if self.args.mode == 'train' or self.args.mode == 'feature_extract': 
            print("Extracting features...")
            feature_extractor = FeatureExtractor(args=self.args)
            feature_extractor.extract_features()
            print("⏳ Initiating UMAP for visualization of your classes prior to training...")
            umap_data = UMAPVisualizer(args=self.args)
            umap_data.reduce_features()
            umap_data.plot_3d_scatter()
            umap_data.create_original_data_frame()
            print("✅ UMAP completed.")
            if self.args.mode == 'train':
                print("⏳Starting model training...")
                from ..training import TrainModel  
                tm = TrainModel(args=self.args)
                tm.clean_data()
                tm.a2000()
                print("Starting hyperparameter tuning...")
                tm.tune_hyperparameters()
                tm.test_model()
                tm.feature_importance()
                print("✅ Model training completed.")
            else:  
                print("✅ ORF feature extraction completed.")
        else :
            print("✅ Pseudo-insilico sequences generation complete.")
        
    def predict(self):
        
        print("⏳Sequences are being fielded...")
        seq_extractor = SequenceExtractor(args=self.args)
        seq_extractor.extract_unknown_sequences()
        print("✅ Sequences fielded.")

        print("⏳Extracting features...")
        feature_extractor = FeatureExtractor(args=self.args)
        feature_extractor.extract_features()
        print("✅ Feature extractions are set and completed.")
        
        print("⏳Throwing features into the prediction algorithm...")
        predictions = smORFPredictor(args=self.args)
        predictions.dansby()
        print("✅ Predictions out and completed!")
        self.__consensus(predictions)
        
    def demo(self):
        print("▶️ You have initiated the demo...")
        
        seq_extractor = SequenceExtractor(args=self.args)
        seq_extractor.create_positive_gtf()
        print("✅ GTF for positive ORFs completed.")
        seq_extractor.extract_sequences()
        print("✅ Amino acids and DNA for each ORF extracted.")

        print("⏳Generating insilico decoy sequences based on your putative smORFs...")
        decoy = NegativeSet(args=self.args)
        decoy.turn_two()
        decoy.combine_databases()
        print("✅ Random decoy sequences generated.")
        
        print("⏳Extracting features...")
        feature_extractor = FeatureExtractor(args=self.args)
        feature_extractor.extract_features()
        print("✅ ORF feature extraction completed.")

        print("⏳Initiating UMAP visualization...")
        umap_data = UMAPVisualizer(args=self.args)
        print("🛑 Early termination of UMAP visualization for demo purposes.")

        print("⏳Starting custom model training...")
        from ..training import TrainModel  
        tm = TrainModel(args=self.args)
        print("🛑 Early termination of hyperparameter tuning for demo purposes.")
        
        print("⏳Throwing features into the standard Miller et al., 2025 ShortStop prediction algorithm...")
        predictions = smORFPredictor(args=self.args)
        predictions.dansby()
        print("✅ Predictions out and completed!")
        self.__consensus(predictions)
        
        #self.__cleanup_output_directory(self.outdir)  # Replace "--outdir" with the actual variable holding the output directory path
        
        print("✅ Demo completed.")
        
    def __consensus(self, predictions):
        if self.args.no_consensus:
            return
        print("⏳Scoring with the 100 consensus models (SAM-Consensus-50/90/100)...")
        try:
            predictions.consensus()
        except RuntimeError as e:
            raise SystemExit(f"❌ SAM-Consensus failed: {e}\n   The standard model predictions were written to "
                             f"{predictions.predictionsDir}.")
        print("✅ SAM-Consensus completed!")

    def __cleanup_output_directory(self, outdir):
        """Removes all directories and files in the specified output directory and recreates the directory."""
        try:
            # Remove the output directory and all its contents
            shutil.rmtree(outdir)
            print(f"🧹 Output directory '{outdir}' cleaned up successfully.")
        except Exception as e:
            print(f"Error cleaning up output directory '{outdir}': {e}")

        