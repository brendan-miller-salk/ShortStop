import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..pipeline import PipelineStructure


class DatabaseCombiner(PipelineStructure):
    def __init__(self, args):
        super().__init__(args=args)
        self.set_train_attributes()
        self.combinedDatabase = None

    def combine(self):
        print("Combining different databases.\n")
        uniprot_smorfs_sequences = pd.read_csv(self.uniprotsmORFsSequences)
        insilico_seqs_df = pd.read_csv(self.insilicoSequencesDF, sep='\t')

        uniprot_smorfs_insilico = uniprot_smorfs_sequences.append(insilico_seqs_df, ignore_index=True)
        self.combinedDatabase = uniprot_smorfs_insilico
        self.combinedDatabase.to_csv(self.combinedDatabaseDF, index=False)

    def plot_sequence_length_distribution(self):
        print("Plotting sequence length distribution.\n")
        sns.set(style="whitegrid")
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        ax = sns.boxplot(x="type", y="length", data=self.combinedDatabase)
        plt.savefig(f'{self.plotsDir}/database_sequence_length_distribution.png', dpi=300)
