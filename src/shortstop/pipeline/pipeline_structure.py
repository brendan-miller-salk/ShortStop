import sys

from ..utils import check_dir, check_multi_dirs


class PipelineStructure:
    def __init__(self, args):
        self.args = args
        
        """
        Defines the output directories.
        """
        
        self.outdir = args.outdir
        self.databaseDir = f'{self.outdir}/sequences'
        self.plotsDir = f'{self.outdir}/plots'
        self.featuresDir = f'{self.outdir}/features'
        self.modelsDir = f'{self.outdir}/models'
        self.predictionsDir = f'{self.outdir}/predictions'
        check_multi_dirs([self.outdir, self.databaseDir, self.plotsDir, self.featuresDir, self.modelsDir, self.predictionsDir])

    def set_train_attributes(self):
        
        """
        Defines the attributes for the train mode.
        """
        
        self.__define_input_files()
        self.__define_output_files()
        

    def set_prediction_attributes(self):
        
        """
        Defines the attributes for the prediction mode.
        """
        
        self.__define_input_files()
        self.__define_output_files()
        
    def __define_input_files(self):
        
        """
        Defines the attributes for the input files provided by the user.
        """

        if self.args.mode == 'train' or self.args.mode == 'pseudo' or self.args.mode == 'feature_extract':
            self.positiveFunction = self.args.positive_functions
            self.positiveDF = self.args.positive_ids
            self.positiveGTF = self.args.positive_gtf
            self.toBePredictedGTF = self.args.putative_smorfs_gtf
            self.genome = self.args.genome
        elif self.args.mode == 'predict':
            self.genome = self.args.genome
            self.toBePredictedGTF = self.args.putative_smorfs_gtf
            self.orfs_features_in_train_model = self.args.orfs_features_in_train_model
            self.orfs_to_be_predicted = self.args.orfs_to_be_predicted
            self.model_scaler = self.args.model_scaler
            self.model = self.args.model
        elif self.args.mode == 'demo':
            self.positiveFunction = self.args.positive_functions
            self.positiveDF = self.args.positive_ids
            self.positiveGTF = self.args.positive_gtf
            self.toBePredictedGTF = self.args.putative_smorfs_gtf
            self.genome = self.args.genome
            self.orfs_features_in_train_model = self.args.orfs_features_in_train_model
            self.orfs_to_be_predicted = self.args.orfs_to_be_predicted
            self.model_scaler = self.args.model_scaler
            self.model = self.args.model

    def __define_output_files(self):
        
        """
        Defines the attributes for the output files generated by the pipeline.
        """
        
        # Define the output files
        self.sequencesWithFunctions = f'{self.databaseDir}/positive_and_unknown_sequences.csv'
        self.positiveMicroproteinsGTF = f"{self.databaseDir}/positive_proteins.gtf"
        self.positive_and_unknown_sequences = f'{self.databaseDir}/positive_and_unknown_sequences.csv'
        self.unknown_sequences = f'{self.databaseDir}/unknown_sequences.csv'
        self.insilicoSequencesDF = f'{self.databaseDir}/insilico_sequences.csv'
        self.combinedDatabaseDF = f'{self.databaseDir}/positive_unknown_insilico_sequences.csv'

        # Define the output files for the features
        self.orfsFeatures = f'{self.featuresDir}/extracted_features_of_smorfs.csv'
        self.umapDF = f'{self.featuresDir}/umap_data.csv'

        # Define the output files for the models
        self.orfs_features_in_train_model = f'{self.modelsDir}/orfs_features_in_train_model.csv'
        self.umapHtml = f"{self.plotsDir}/umap_3d_scatter_reduced_features.html"
