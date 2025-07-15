import time
from collections import Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import umap

from ..pipeline import PipelineStructure


class UMAPVisualizer(PipelineStructure):
    def __init__(self, args):
        super().__init__(args=args)
        self.set_train_attributes()

        self.reducedFeatures = None

    def reduce_features(self):
        
            """
            Reduces the features using UMAP algorithm and stores the reduced features in self.reducedFeatures.
            """
            
            features = pd.read_csv(self.orfsFeatures)

            orf_id = features['orf_id']
            type = features['type']
            local = features['local']
            label = features['label']
            self.labels = label

            # Drop orf_id, type, and local to just include float values
            features = features.drop(['orf_id', 'local', 'type', 'label'], axis=1)

            features = features.loc[:, ~features.columns.str.startswith('cds')]

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            time_start = time.time()
            umap_model = umap.UMAP(n_components=3, n_neighbors= 100)  # Specify the desired number of components as 3 for 3D
            umap_result = umap_model.fit_transform(scaled_features)

            reduced_features = pd.DataFrame(umap_result)
            reduced_features['orf_id'] = orf_id
            reduced_features['type'] = type
            reduced_features['local'] = local

            reduced_features = reduced_features.rename(columns={0: "umap_0", 1: "umap_1", 2: "umap_2"})
            reduced_features = reduced_features[reduced_features.local != "Missing"] #Remove ORFs with missing cellular compartment annotation
            self.reducedFeatures = reduced_features

            print(f"UMAP took {time.time() - time_start:.2f} seconds")

    def plot_3d_scatter(self):
        fig = px.scatter_3d(self.reducedFeatures, x="umap_0", y="umap_1", z="umap_2", color="local",
                            color_discrete_map={"Cytoplasm": "yellow", "Secreted": "blue", "Random": "black",
                                                "ToBePredicted": "red"})
        fig.update_traces(marker={'size': 3, 'opacity': 1})
        fig.update_layout(hovermode="y")
        fig.write_html(self.umapHtml, include_plotlyjs=True, full_html=True)

    def create_original_data_frame(self):
        df = self.reducedFeatures
        df = df[df.local != "Missing"]
        df.to_csv(self.umapDF, index=False)