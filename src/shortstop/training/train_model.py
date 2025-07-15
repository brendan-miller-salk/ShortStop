import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, accuracy_score
)
import eli5
from eli5.sklearn import PermutationImportance
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier

from ..pipeline import PipelineStructure

class TrainModel(PipelineStructure):
    def __init__(self, args):
        super().__init__(args=args)
        self.set_train_attributes()
        self.labels = None
        self.dnaData = None
        self.aaData = None
        self.orfIds = None
        self.data = None
        self.labelTrain = None
        self.dnaTest = None
        self.aaTest = None
        self.labelTest = None
        self.orfTrain = None
        self.orfTest = None
        self.history = None
        self.best_model = None
        self.scaler = None

    def fit_transform_scaler(self, train_data):
        
        """Fit the scaler with training data and transform it."""
        
        self.scaler = StandardScaler().fit(train_data)
        return self.scaler.transform(train_data)

    def transform_data(self, data):
        
        """Transform data with the already fitted scaler."""
        
        return self.scaler.transform(data)

    def clean_data(self):
        
        """
        Cleans the data by removing unnecessary columns and filtering out certain labels.
        """
        
        features = pd.read_csv(self.orfsFeatures)
        features = features[features.local != 'Unknown']
        features = features.drop(features.filter(regex='cds|local|type').columns, axis=1)

        labels = pd.read_csv(self.umapDF)
        labels = labels[labels.local != 'ToBePredicted']
        labels = labels[labels.local != 'Missing']

        df = pd.merge(features, labels, on='orf_id')

        self.dnaData = df.loc[:, df.columns.str.startswith(('5_prime', '3_prime', 'kozak', 'first_50'))].to_numpy()
        np.savetxt(f"{self.modelsDir}/dna_columns.txt", df.loc[:, df.columns.str.startswith(('5_prime', '3_prime', 'kozak', 'first_50'))].columns, fmt='%s')
        aa_columns = [col for col in df if not any(prefix in col for prefix in ('3_prime', '5_prime', 'kozak', 'label', 'orf_id', 'cds', 'first_50', 'type', 'local', 'umap_0', 'umap_1', 'umap_2'))]
        np.savetxt(f"{self.modelsDir}/aa_columns.txt", df[aa_columns].columns, fmt='%s')
        self.aaData = df[aa_columns].to_numpy()
        self.orfIds = df['orf_id'].values
        
        dna_names = df.loc[:, df.columns.str.startswith(('5_prime', '3_prime', 'kozak', 'first_50'))].columns
        aa_names = df[aa_columns].columns
        self.dna_aa_names = np.concatenate((dna_names, aa_names))

        mapping = {"Random": 0, "Cytoplasm": 1, "Secreted": 2}
        df['label_encoded'] = df['local'].map(mapping)
        df = df.dropna(subset=['label_encoded'])  # adjust the column name as necessary
        self.labels = df['label_encoded'].values
        print(len(self.dnaData), len(self.aaData), len(self.labels), len(self.orfIds))

    def a2000(self):
        
        """
        Split the data into training and testing sets, concatenate the DNA and AA features,
        transform the data using a scaler, and save the scaler object.
        """
        
        dna_train, self.dnaTest, aa_train, self.aaTest, self.labelTrain, self.labelTest, self.orfTrain, self.orfTest = train_test_split(
            self.dnaData, self.aaData, self.labels, self.orfIds, test_size=0.2, stratify=self.labels)

        train_data = np.concatenate((dna_train, aa_train), axis=1)
        
        # Save training data
        np.savetxt(f"{self.modelsDir}/train_data.txt", train_data)
        
        print(f"Number of DNA features: {dna_train.shape[1]}")
        print(f"Number of AA features: {aa_train.shape[1]}")
        self.data = self.fit_transform_scaler(train_data)
        joblib.dump(self.scaler, f"{self.modelsDir}/scaler.save")

    def tune_hyperparameters(self):
        
        """
        Train a neural network and tune hyperparameters using GridSearchCV.
        """
        
        # Define the model and the hyperparameters to tune
        def create_model(data_shape):
            def model_fn(nodes_layer_one=512, nodes_layer_two=128, dropout_rate=0.4, learning_rate=0.0001):
                input_layer = Input(shape=(data_shape,))
                x = Dropout(dropout_rate)(Activation('relu')(BatchNormalization()(Dense(nodes_layer_one)(input_layer))))
                if nodes_layer_two > 0:
                    x = Dropout(dropout_rate)(Activation('relu')(BatchNormalization()(Dense(nodes_layer_two)(x))))
                output_layer = Dense(3, activation='softmax')(x)
                model = Model(inputs=input_layer, outputs=output_layer)
                model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
                return model
            return model_fn

        model = KerasClassifier(build_fn=create_model(self.data.shape[1]), verbose=0)
        param_grid = {
            'nodes_layer_one': [256, 512],
            'nodes_layer_two': [0],
            'dropout_rate': [0.2],
            'validation_split': [0.2],
            'learning_rate': [0.001, 0.00001],
            'batch_size': [48, 96],
            'epochs': [200],
            'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        }
        
        # Tune hyperparameters
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        labels_encoded = to_categorical(self.labelTrain)
        grid_result_nn = grid.fit(self.data, labels_encoded)
        
        # Get the best model
        nn_best_score = grid_result_nn.best_score_
        nn_best_model = grid_result_nn.best_estimator_.model
        print('Neural Network - Best score:', nn_best_score)
        print('Neural Network - Best params:', grid_result_nn.best_params_)
    
        # Save the model history
        self.history = pd.DataFrame(nn_best_model.history.history)
        self.history.to_csv(f"{self.modelsDir}/model_history.csv", index=False)

        # Plot the model architecture
        self.modelArchitecturePlot = f"{self.plotsDir}/nnet_model_architecture.png"
        plot_model(nn_best_model, show_shapes=True, to_file=self.modelArchitecturePlot)

        # Plot the training metrics
        self.modelAccuracyPlot = f"{self.plotsDir}/nnet_model_accuracy.png"
        plt.clf()
        plt.plot(self.history['accuracy'], label='train accuracy')
        plt.plot(self.history['val_accuracy'], label='val accuracy')
        plt.plot(self.history['loss'], label='train loss')
        plt.plot(self.history['val_loss'], label='val loss')
        plt.title('Model accuracy and loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.legend(loc='best')
        plt.savefig(self.modelAccuracyPlot, dpi=300)
        
        # Save nn_best_model
        nn_best_model.save(f"{self.modelsDir}/best_nn_model.h5")
        self.nn_best_model = nn_best_model
        
        
        """
        Tune a XGBoost model using GridSearchCV 
        """
        
        # Define the model and the hyperparameters to tune
        xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, use_label_encoder=False)
        
        param_grid = {
        'n_estimators': [250, 500],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5, 1.0]
        }

        # Tune hyperparameters
        grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result_xgb = grid.fit(self.data, self.labelTrain)
        
        # Get the best model
        xgb_best_score = grid_result_xgb.best_score_
        xgb_best_model = grid_result_xgb.best_estimator_
        print('XGBoost - Best score:', xgb_best_score)
        print('XGBoost - Best params:', grid_result_xgb.best_params_)
        
        # Save best XGB model
        xgb_best_model.save_model(f"{self.modelsDir}/best_xgb_model.model")
        self.xgb_best_model = xgb_best_model 
        
        """
        Tune a Random Forest using GridSearchCV 
        """
        
        rf_model = RandomForestClassifier()
        
        param_grid = {
            'n_estimators': [100, 250],
            'max_depth': [3, 9],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True]
        }
        
        grid = GridSearchCV(estimator=rf_model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result_rf = grid.fit(self.data, self.labelTrain)
        
        # Get the best model
        rf_best_score = grid_result_rf.best_score_
        rf_best_model = grid_result_rf.best_estimator_
        print('Random Forest - Best score:', rf_best_score)
        print('Random Forest - Best params:', grid_result_rf.best_params_)
        
        # Save best RF model
        joblib.dump(rf_best_model, f"{self.modelsDir}/best_rf_model.pkl")
        self.rf_best_model = rf_best_model
        
    
    def test_model(self):
        
        # Create the test data
        test_data = np.concatenate((self.dnaTest, self.aaTest), axis=1)
        test_data = self.transform_data(test_data)
        self.test_data = test_data
        
        # Save test data
        np.savetxt(f"{self.modelsDir}/test_data.txt", test_data)

        # Binarize the labels for multi-class ROC curve
        n_classes = 3
        self.labelTest = label_binarize(self.labelTest, classes=[0, 1, 2])
        
        # Predict probabilities
        nn_probs = self.nn_best_model.predict(test_data)
        xgb_probs = self.xgb_best_model.predict_proba(test_data)
        rf_probs = self.rf_best_model.predict_proba(test_data)

        #  Initialize lists to store AUCs
        nn_auc = []
        xgb_auc = []
        rf_auc = []

        # Initialize dictionaries to store FPR and TPR data
        nn_data = {}
        xgb_data = {}
        rf_data = {}

        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(self.labelTest[:, i], nn_probs[:, i])
            nn_auc.append(auc(fpr, tpr))
            nn_data[f'Class_{i}'] = {'fpr': fpr, 'tpr': tpr}
            plt.plot(fpr, tpr, color=color, linestyle='-', label=f'NN Class {i} (AUC = {nn_auc[-1]:0.2f})')
        

            fpr, tpr, _ = roc_curve(self.labelTest[:, i], xgb_probs[:, i])
            xgb_auc.append(auc(fpr, tpr))
            xgb_data[f'Class_{i}'] = {'fpr': fpr, 'tpr': tpr}
            plt.plot(fpr, tpr, color=color, linestyle='--', label=f'XGB Class {i} (AUC = {xgb_auc[-1]:0.2f})')

            fpr, tpr, _ = roc_curve(self.labelTest[:, i], rf_probs[:, i]) 
            rf_auc.append(auc(fpr, tpr))
            rf_data[f'Class_{i}'] = {'fpr': fpr, 'tpr': tpr}
            plt.plot(fpr, tpr, color=color, linestyle=':', label=f'RF Class {i} (AUC = {rf_auc[-1]:0.2f})')

       # Function to save raw data to CSV file
        def save_roc_data(clf_name, clf_data):
            all_data = []
            for class_name, data in clf_data.items():
                class_data = pd.DataFrame({
                    'fpr': data['fpr'],
                    'tpr': data['tpr'],
                    'class': class_name,
                    'model': clf_name
                })
                all_data.append(class_data)
            all_data = pd.concat(all_data)
            all_data['iteration'] = np.arange(1, len(all_data) + 1)
            
            return all_data
        
        # Save data for each model separately
        nn_data = save_roc_data('nn', nn_data)
        xgb_data = save_roc_data('xgb', xgb_data)
        rf_data = save_roc_data('rf', rf_data)
        all_data = pd.concat([nn_data, xgb_data, rf_data])
        all_data.to_csv(f"{self.modelsDir}/roc_data.csv", index=False)

        nn_macro_auc = np.mean(nn_auc)
        xgb_macro_auc = np.mean(xgb_auc)
        rf_macro_auc = np.mean(rf_auc)
        print(f'Neural Network Macro-Average ROC AUC: {nn_macro_auc}')
        print(f'XGBoost Macro-Average ROC AUC: {xgb_macro_auc}')
        print(f'Random Forest Macro-Average ROC AUC: {rf_macro_auc}')
        
        # Save the macro-average ROC AUC as a csv with first column as model and second column as macro-average ROC AUC
        macro_auc_df = pd.DataFrame({'model': ['nn', 'xgboost', 'rf'], 'macro_auc': [nn_macro_auc, xgb_macro_auc, rf_macro_auc]})
        macro_auc_df.to_csv(f"{self.modelsDir}/macro_auc.csv", index=False)

        # Choose the best model based on the macro-average ROC AUC
        if nn_macro_auc > xgb_macro_auc and nn_macro_auc > rf_macro_auc:
            best_model_path = f"{self.modelsDir}/best_nn_model.h5"
        elif xgb_macro_auc > nn_macro_auc and xgb_macro_auc > rf_macro_auc:
            best_model_path = f"{self.modelsDir}/best_xgb_model.model"
        else:
            best_model_path = f"{self.modelsDir}/best_rf_model.pkl"
        self.best_model_path = best_model_path
        
        print(f'The best model is: {best_model_path}')
        
        
        # Plot the ROC curve
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Comparison')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.plotsDir}/roc_curve_comparison.png", dpi=300)
        
        # Get F1 score, precision, recall, and accuracy
        if best_model_path.endswith('.model'):
            model = xgb.XGBClassifier()
            model.load_model(best_model_path)
            predictions = model.predict(test_data)
        else:
            model = joblib.load(best_model_path)
            predictions = model.predict(test_data)
        
        # Create confusion matrix
        cm = confusion_matrix(self.labelTest, predictions)   
        f1 = f1_score(self.labelTest, predictions)
        precision = precision_score(self.labelTest, predictions)
        recall = recall_score(self.labelTest, predictions)
        accuracy = accuracy_score(self.labelTest, predictions)
        
        # Save confusion matrix and metrics
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(f"{self.modelsDir}/confusion_matrix.csv", index=False)
        metrics_df = pd.DataFrame({'f1': [f1], 'precision': [precision], 'recall': [recall], 'accuracy': [accuracy]})
        metrics_df.to_csv(f"{self.modelsDir}/metrics.csv", index=False)
        
        return best_model_path
    
    def feature_importance(self):
        
        """
        Will do feature importance on the best nnet or xbg.
        """

        if self.best_model_path.endswith('.model'):
            model = xgb.XGBClassifier() 
            model.load_model(self.best_model_path)  
            importances = model.feature_importances_
            feature_names = self.dna_aa_names 
            feature_importances = pd.DataFrame(importances, index=feature_names, columns=['importance'])
            feature_importances = feature_importances.sort_values(by='importance', ascending=False)
            feature_importances.to_csv(f"{self.modelsDir}/xgb_feature_importances.csv")
            # Plot the first 20 features
            plt.figure()
            plt.bar(feature_importances.index[:20], feature_importances['importance'][:20])
            plt.xticks(rotation=90)
            plt.title('XGBoost Feature Importance')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(f"{self.plotsDir}/xgb_feature_importance.png", dpi=300)
            
        else:
            model = joblib.load(self.best_model_path)
            importances = model.feature_importances_
            feature_names = self.dna_aa_names
            feature_importances = pd.DataFrame(importances, index=feature_names, columns=['importance'])
            feature_importances = feature_importances.sort_values(by='importance', ascending=False)
            feature_importances.to_csv(f"{self.modelsDir}/rf_feature_importances.csv")
            # Plot the first 20 features
            plt.figure()
            plt.bar(feature_importances.index[:20], feature_importances['importance'][:20])
            plt.xticks(rotation=90)
            plt.title('Random Forest Feature Importance')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(f"{self.plotsDir}/rf_feature_importance.png", dpi=300)