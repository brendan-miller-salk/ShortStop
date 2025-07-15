from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
 

from ..pipeline import PipelineStructure
from ..utils import check_dir

class smORFPredictor(PipelineStructure):
    def __init__(self, args):
        super().__init__(args=args)
        self.set_prediction_attributes()
        
    def align_and_confirm_features(self):
        self.orfs_features_in_train_model = pd.read_csv(self.args.orfs_features_in_train_model)
        self.orfs_to_be_predicted = pd.read_csv(self.args.orfs_to_be_predicted)
        
        missing_features = set(self.orfs_features_in_train_model.columns) - set(self.orfs_to_be_predicted.columns)
        
        missing_features_df = pd.DataFrame(0.0, index=self.orfs_to_be_predicted.index, columns=list(missing_features), dtype=float)
        
        self.orfs_to_be_predicted = pd.concat([self.orfs_to_be_predicted, missing_features_df], axis=1)
        
        # Drop features not present in the train model and ensure column order matches the train model
        missing_features_two = set(self.orfs_to_be_predicted.columns) - set(self.orfs_features_in_train_model.columns)
        self.orfs_to_be_predicted = self.orfs_to_be_predicted.drop(columns=missing_features_two)
        self.orfs_to_be_predicted = self.orfs_to_be_predicted[self.orfs_features_in_train_model.columns]
        
        # Make a copy to defragment the DataFrame
        self.orfs_to_be_predicted = self.orfs_to_be_predicted.copy()

        # Confirmation message
        print("Confirmed: Your data is appropriately scaled based on the original model's train platform.")


        
    def dansby(self):
        self.align_and_confirm_features()
        orf_ids = self.orfs_to_be_predicted['orf_id']
                
        if self.model.endswith('.h5'): # Neural Net
            model = tf.keras.models.load_model(self.model)
            self.__scaler() 
            predictions = model.predict(self.data) 
            print(predictions)
        elif self.model.endswith('.model'):
            model = xgb.XGBClassifier() 
            model.load_model(self.model)
            self.__scaler()
            predictions = model.predict_proba(self.data)
            print(predictions)
        elif self.model.endswith('.pkl'):
            model = joblib.load(self.model)
            self.__scaler()
            predictions = model.predict_proba(self.data)
            print(predictions)

        predictions_df = pd.DataFrame(predictions, columns=['prism_probability', 'intracellular', 'extracellular_secreted'])
        predictions_df['sam_probability'] = predictions_df['intracellular'] + predictions_df['extracellular_secreted']
        predictions_df['orf_id'] = orf_ids
        predictions_df = predictions_df[['orf_id','prism_probability', 'sam_probability']]
        predictions_df.to_csv(f'{self.predictionsDir}/sams.csv', index=False)

        labels = np.argmax(predictions, axis=1)
        percentages = np.max(predictions, axis=1)

        # Map integer labels to string class names BEFORE creating the DataFrame
        label_names = np.array([
            'prisms' if l == 0 else
            'sam_intracellular' if l == 1 else
            'sam_secreted' for l in labels
        ])

        predicted_classes = pd.DataFrame({
            'orf_id': orf_ids,
            'classification': label_names,
            'probability': percentages
        })
    
        predicted_classes = predicted_classes.sort_values(by=['probability'], ascending=False)
        # Save the predicted classes to a CSV file
        predicted_classes.to_csv(f'{self.predictionsDir}/shortstop_classifications.csv', index=False)

        for class_name in ['prisms', 'sam_intracellular', 'sam_secreted']:
            class_data = predicted_classes[predicted_classes['classification'] == class_name]
            class_data.to_csv(f'{self.predictionsDir}/{class_name}.csv', index=False)
        
    
    def __scaler(self):
        self.scaler = joblib.load(self.args.model_scaler)
        
        self.__aa_split()
        self.__dna_split()
        
        print(f"Number of DNA features being used to predict: {self.dna_data.shape[1]}")
        print(f"Number of AA features being used to predict: {self.aa_data.shape[1]}")
        
        self.data = np.concatenate((self.dna_data, self.aa_data), axis=1)
        
        self.data =self.scaler.transform(self.data)
        
    def __dna_split(self):
        self.dna_data = self.orfs_to_be_predicted.loc[:, self.orfs_to_be_predicted.columns.str.startswith('5_prime') | self.orfs_to_be_predicted.columns.str.startswith('3_prime') | self.orfs_to_be_predicted.columns.str.startswith('kozak') | self.orfs_to_be_predicted.columns.str.startswith('first_50')]
        self.dna_data = self.dna_data.to_numpy()
        
    def __aa_split(self):
        self.aa_data = self.orfs_to_be_predicted.drop(self.orfs_to_be_predicted .filter(regex='3_prime').columns, axis=1)
        self.aa_data = self.aa_data.drop(self.aa_data.filter(regex='5_prime').columns, axis=1)
        self.aa_data = self.aa_data.drop(self.aa_data.filter(regex='kozak').columns, axis=1)
        self.aa_data = self.aa_data.drop(self.aa_data.filter(regex='first_50').columns, axis=1)
        self.aa_data = self.aa_data.drop(self.aa_data.filter(regex='label').columns, axis=1)
        self.aa_data = self.aa_data.drop(self.aa_data.filter(regex='type').columns, axis=1)
        self.aa_data = self.aa_data.drop(self.aa_data.filter(regex='orf_id').columns, axis=1)
        self.aa_data = self.aa_data.drop(self.aa_data.filter(regex='local').columns, axis=1)
        self.aa_data = self.aa_data.to_numpy()