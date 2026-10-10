from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


from ..pipeline import PipelineStructure
from ..utils import check_dir

DNA_PREFIXES = ('5_prime', '3_prime', 'kozak', 'first_50')
AA_DROP = ['3_prime', '5_prime', 'kozak', 'first_50', 'label', 'type', 'orf_id', 'local']
SAM_PROB = 0.5                 # each consensus model votes SAM when P(intracellular) + P(secreted) > 0.5
CONSENSUS_LEVELS = (50, 90, 100)  # sam_consensus_50: > 50% of models; _90: >= 90%; _100: all models


def align_features(features, train_columns):
    """Add missing training features as 0, drop extras, and match the training column order."""
    missing_features = set(train_columns) - set(features.columns)
    missing_features_df = pd.DataFrame(0.0, index=features.index, columns=list(missing_features), dtype=float)
    aligned = pd.concat([features, missing_features_df], axis=1)

    # Drop features not present in the train model and ensure column order matches the train model
    missing_features_two = set(aligned.columns) - set(train_columns)
    aligned = aligned.drop(columns=missing_features_two)
    aligned = aligned[train_columns]

    # Make a copy to defragment the DataFrame
    return aligned.copy()


def model_matrix(aligned, scaler_path):
    """DNA features then amino-acid features, scaled with the model's own scaler."""
    dna_data = aligned.loc[:, aligned.columns.str.startswith(DNA_PREFIXES)].to_numpy()
    aa_data = aligned
    for pattern in AA_DROP:
        aa_data = aa_data.drop(aa_data.filter(regex=pattern).columns, axis=1)
    data = np.concatenate((dna_data, aa_data.to_numpy()), axis=1)
    return joblib.load(scaler_path).transform(data)


class smORFPredictor(PipelineStructure):
    def __init__(self, args):
        super().__init__(args=args)
        self.set_prediction_attributes()

    def align_and_confirm_features(self):
        self.orfs_features_in_train_model = pd.read_csv(self.args.orfs_features_in_train_model)
        self.orfs_to_be_predicted = pd.read_csv(self.args.orfs_to_be_predicted)
        self.orfs_to_be_predicted = align_features(self.orfs_to_be_predicted, self.orfs_features_in_train_model.columns)


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

        # Kept in input order for the consensus table
        self.standard = pd.DataFrame({'orf_id': orf_ids, 'classification': label_names,
                                      'sam_probability': predictions_df['sam_probability']})

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

    def consensus(self):
        """Score the smORFs with every consensus model and write the consensus table.

        Each model votes SAM when P(intracellular) + P(secreted) > 0.5. sam_consensus_percentage is the
        share of models voting SAM; sam_consensus_50 / _90 / _100 flag > 50%, >= 90% and 100% of models.
        """
        from ..consensus import ensure_consensus_models, MANIFEST

        consensus_dir = ensure_consensus_models(self.args.consensus_dir)
        manifest = pd.read_csv(consensus_dir / MANIFEST)
        features = pd.read_csv(self.args.orfs_to_be_predicted)

        probs = np.empty((len(manifest), len(features), 3), dtype=np.float32)
        cached_columns, aligned = None, None
        for i, model_id in enumerate(manifest['model_id']):
            model_dir = consensus_dir / model_id
            columns = list(pd.read_csv(model_dir / 'orfs_features_in_train_model.csv', nrows=0).columns)
            if columns != cached_columns:  # one aligned matrix at a time keeps memory flat
                cached_columns, aligned = columns, align_features(features, columns)
            model = xgb.XGBClassifier()
            model.load_model(str(model_dir / 'best_xgb_model.model'))
            probs[i] = model.predict_proba(model_matrix(aligned, model_dir / 'scaler.save'))
            if (i + 1) % 25 == 0:
                print(f"   {i + 1}/{len(manifest)} consensus models scored")

        table, summary = consensus_table(probs, features['orf_id'], self.standard, manifest)
        check_dir(self.consensusDir)
        table.to_csv(f'{self.consensusDir}/shortstop_consensus.csv', index=False)
        with open(f'{self.consensusDir}/consensus_summary.txt', 'w') as f:
            f.write(summary)
        print(summary)

    def __scaler(self):
        self.data = model_matrix(self.orfs_to_be_predicted, self.args.model_scaler)


def consensus_table(probs, orf_ids, standard, manifest):
    """Per-smORF consensus table and a plain-text summary from (models x smORFs x 3) probabilities."""
    n_models = probs.shape[0]
    sam_prob = probs[:, :, 1:].sum(axis=2)
    vote = sam_prob > SAM_PROB
    n_sam = vote.sum(axis=0)
    percentage = 100 * n_sam / n_models
    levels = {50: 2 * n_sam > n_models, 90: 10 * n_sam >= 9 * n_models, 100: n_sam == n_models}  # exact integer tests
    n_intra = (vote & (probs[:, :, 1] >= probs[:, :, 2])).sum(axis=0)
    n_secreted = n_sam - n_intra
    standard_sam = standard['classification'].to_numpy() != 'prisms'

    consensus = pd.DataFrame({
        'orf_id': orf_ids.to_numpy(),
        'classification': standard['classification'].to_numpy(),
        'sam_probability': standard['sam_probability'].to_numpy(),
        'sam': standard_sam,
        'sam_consensus_percentage': percentage.round(1),
        **{f'sam_consensus_{level}': levels[level] for level in CONSENSUS_LEVELS},
        'n_models_sam': n_sam,
        'mean_sam_probability': sam_prob.mean(axis=0).round(4),
        'sd_sam_probability': sam_prob.std(axis=0).round(4),
        'consensus_class': np.where(n_sam == 0, '', np.where(n_intra >= n_secreted, 'sam_intracellular', 'sam_secreted')),
        'class_agreement': np.where(n_sam == 0, np.nan, np.maximum(n_intra, n_secreted) / np.maximum(n_sam, 1)).round(3),
    }).sort_values(['sam_consensus_percentage', 'mean_sam_probability'], ascending=False)

    # Agreement between models on this input
    X = vote.astype(np.float32)
    inter = X @ X.T
    size = X.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    jaccard = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)[np.triu_indices(n_models, 1)]
    pairs = (n_sam * (n_sam - 1) + (n_models - n_sam) * (n_models - n_sam - 1)) / (n_models * (n_models - 1))
    p_sam = vote.mean()
    p_chance = p_sam ** 2 + (1 - p_sam) ** 2
    kappa = (pairs.mean() - p_chance) / (1 - p_chance) if p_chance < 1 else float('nan')

    def count(sel):
        intra = (sel & (n_intra >= n_secreted)).sum()
        return f"{sel.sum():>6d}   ({intra} intracellular, {sel.sum() - intra} secreted by consensus majority)"

    lines = [
        f"ShortStop SAM-Consensus: {n_models} models, {len(orf_ids)} smORFs",
        f"Each model votes SAM when P(intracellular) + P(secreted) > {SAM_PROB}",
        "",
        f"  SAM (standard model)   {standard_sam.sum():>6d}",
        f"  SAM-Consensus-50       {count(levels[50])}   > 50% of models",
        f"  SAM-Consensus-90       {count(levels[90])}   >= 90% of models",
        f"  SAM-Consensus-100      {count(levels[100])}   all models",
        f"  SAM and SAM-Consensus-50: {(standard_sam & levels[50]).sum()}; SAM only: {(standard_sam & ~levels[50]).sum()}; "
        f"SAM-Consensus-50 only: {(~standard_sam & levels[50]).sum()}",
        "",
        f"SAM votes per model: median {np.median(size):.0f}, range {size.min():.0f}-{size.max():.0f}",
        f"Pairwise overlap of SAM lists (Jaccard): mean {jaccard.mean():.3f}, "
        f"5-95% {np.percentile(jaccard, 5):.3f}-{np.percentile(jaccard, 95):.3f}",
        f"Fleiss' kappa (SAM vs not): {kappa:.3f}",
    ]
    if 'is_standard' in manifest and manifest['is_standard'].any():
        i = int(np.flatnonzero(manifest['is_standard'].to_numpy())[0])
        diff = np.abs(sam_prob[i] - standard['sam_probability'].to_numpy()).max()
        lines.append(f"Check: {manifest['model_id'].iloc[i]} reproduces the standard model "
                     f"(max |dP(SAM)| = {diff:.1e})")
    return consensus, "\n".join(lines) + "\n"
