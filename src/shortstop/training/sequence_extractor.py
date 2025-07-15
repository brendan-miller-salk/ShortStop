import sys
import re
import pandas as pd
from Bio import SeqIO

from ..utils import check_dir
from ..pipeline import PipelineStructure
from ..converters import GTFtoSeq


class SequenceExtractor(PipelineStructure):
    def __init__(self, args):
        super().__init__(args)
        self.set_train_attributes()

    def create_positive_gtf(self):
        
        """
        Creates a positive GTF file by filtering a GTF file based on a list of Ensembl gene IDs.

        This method performs the following steps:
        1. Reads a CSV file containing positive ORFs and extracts the Ensembl gene IDs.
        2. Filters a GTF file based on the extracted gene IDs.
        3. Renames the columns in the filtered GTF file.
        4. Saves the filtered GTF file.
        """
        
        positive_orfs = pd.read_csv(self.positiveDF)
        # If the first column is not named "orf_id", then rename it to "orf_id"
        if positive_orfs.columns[0] != "orf_id":
            positive_orfs = positive_orfs.rename(columns={positive_orfs.columns[0]: "orf_id"})
        # If the first column is not named "Ensembl", then rename it to "Ensembl"
        if positive_orfs.columns[1] != "Ensembl":
            positive_orfs = positive_orfs.rename(columns={positive_orfs.columns[1]: "Ensembl"})
        gene_list = positive_orfs["Ensembl"].tolist()
        # Remove floats from the gene_list
        gene_list = [x for x in gene_list if not isinstance(x, float)]
        # Keep only strings that start with "ENST" in gene_list
        gene_list = [x for x in gene_list if str(x).startswith("ENST")]
        # Split each element in gene_list using semi-colon (;) as delimiter
        gene_list = [x.split(";") for x in gene_list]
        # Flatten the list of lists
        gene_list = [item for sublist in gene_list for item in sublist]
        # Remove empty strings from gene_list
        gene_list = [x for x in gene_list if x]
        # Split each element in gene_list using space character as delimiter and keep only the first part
        gene_list = [x.split(" ")[0] for x in gene_list]
        # Load GTF positive_orfs file into a pandas dataframe
        gtf_positive_orfs = pd.read_csv(self.positiveGTF, sep="\t", header=None)
        # Filter gtf by gene_list
        gtf_positive_orfs = gtf_positive_orfs[gtf_positive_orfs[8].str.contains('|'.join(gene_list))]
        # Change gene_id to gene_name
        gtf_positive_orfs[8] = gtf_positive_orfs[8].str.replace("gene_id", "gene_name_id")
        # Change transcript_id to gene_id
        gtf_positive_orfs[8] = gtf_positive_orfs[8].str.replace("transcript_id", "gene_id")
        # Change gene_name_id to transcript_id
        gtf_positive_orfs[8] = gtf_positive_orfs[8].str.replace("gene_name_id", "transcript_id")
        # Save GTF
        gtf_positive_orfs.to_csv(self.positiveMicroproteinsGTF, sep="\t", header=False, index=False)
    
    def extract_unknown_sequences(self):
        
        """
        Extracts unknown sequences from the given GTF and FASTA files.
        
        """
        unknown_orfs = GTFtoSeq(gtf_file=self.toBePredictedGTF, fasta_file=self.genome, cds_order="Last", utr_length=self.args.utr_length)
        unknown_orfs_df = unknown_orfs.extract_sequences()
        unknown_orfs_df['length'] = unknown_orfs_df['aa_seq'].str.len()
        unknown_orfs_df = unknown_orfs_df[unknown_orfs_df['length'] >= 9]
        unknown_orfs_df = unknown_orfs_df[unknown_orfs_df['length'] <= 150]
        unknown_orfs_df['type'] = 'unknown_orfs'
        unknown_orfs_df.to_csv(self.unknown_sequences, index=False)

    def extract_sequences(self):
   
        unknown_orfs = GTFtoSeq(gtf_file=self.toBePredictedGTF, fasta_file=self.genome, cds_order="Last", utr_length=self.args.utr_length)
        unknown_orfs_df = unknown_orfs.extract_sequences()
        unknown_orfs_df['type'] = 'unknown_orfs'
        unknown_orfs_df["transcript_id"] = unknown_orfs_df["orf_id"]
        unknown_orfs_df.to_csv(self.unknown_sequences, index=False)

        positive_orfs = GTFtoSeq(gtf_file=self.positiveMicroproteinsGTF, fasta_file=self.genome, cds_order='First', utr_length=self.args.utr_length)
        positive_orfs_df = positive_orfs.extract_sequences()
        positive_orfs_df['type'] = 'positive_orfs'

        positive_orfs_df = positive_orfs_df.rename(columns={"orf_id": "transcript_id"})
        positive_orfs_df["transcript_id"] = positive_orfs_df["transcript_id"].str.replace('"', '')
        positive_orfs_df = self.__append_ids_to_ensembl_transcripts(positive_orfs_df=positive_orfs_df)
        positive_orfs_unknown_orfs_sequences = pd.concat([positive_orfs_df, unknown_orfs_df], ignore_index=True)
        positive_orfs_unknown_orfs_sequences['length'] = positive_orfs_unknown_orfs_sequences['aa_seq'].str.len()
        positive_orfs_unknown_orfs_sequences = positive_orfs_unknown_orfs_sequences[positive_orfs_unknown_orfs_sequences['length'] >= 9]
        positive_orfs_unknown_orfs_sequences = positive_orfs_unknown_orfs_sequences[positive_orfs_unknown_orfs_sequences['length'] <= 150]
        self.__add_labels_for_propagation(positive_orfs_unknown_orfs_sequences=positive_orfs_unknown_orfs_sequences)
        

    def __append_ids_to_ensembl_transcripts(self, positive_orfs_df):
        # First, get the positive_orfs Entry IDs
        positive_orfs = pd.read_csv(self.positiveDF)
        # Create a new row for each Ensembl id that has a semi-colon (;) in it, and then add the Ensembl id to the new row
        positive_orfs = positive_orfs.assign(Ensembl=positive_orfs.Ensembl.str.split(';')).explode('Ensembl')
        # Remove floats friom Ensembl column
        positive_orfs['Ensembl'] = positive_orfs['Ensembl'].astype(str)
        # Keep rows that start with ENST
        positive_orfs = positive_orfs[positive_orfs.Ensembl.str.startswith('ENST')]
        # Remove empty rows
        positive_orfs = positive_orfs[positive_orfs.Ensembl != 'nan']
        # Remove all text after the first space in the Ensembl column
        positive_orfs["Ensembl"] = positive_orfs["Ensembl"].str.split(" ").str[0]
        # Change Ensembl column name to transcript_id
        positive_orfs = positive_orfs.rename(columns={'Ensembl': 'transcript_id'})
        # Convert positive_orfs to a data frame
        positive_orfs_names = pd.DataFrame(positive_orfs)
        # Second, merge the positive_orfs Entry names with the sequence data frame that contains the Ensemble IDs
        positive_orfs_df = pd.merge(positive_orfs_df, positive_orfs_names, on='transcript_id', how='left')
        return positive_orfs_df


    def __add_labels_for_propagation(self, positive_orfs_unknown_orfs_sequences):
        
        functions = pd.read_csv(self.positiveFunction)
        sequences_with_functions = positive_orfs_unknown_orfs_sequences.merge(functions, how='left', on='orf_id')
        sequences_with_functions["function"] = sequences_with_functions["function"].fillna('')
        sequences_with_functions["local"] = "Missing"
        for index, row in sequences_with_functions.iterrows():
            function = row['function']
            function = re.sub(r'[^a-zA-Z]', '', function)
            if re.search(r"(secrete)", function, flags=re.IGNORECASE):  
                sequences_with_functions.at[index, 'local'] = "Secreted"
            elif re.search(r"(cytoplasm|mitochond|nuc|golgi|endoplas|membrane)", function, flags=re.IGNORECASE):
                sequences_with_functions.at[index, 'local'] = "Cytoplasm"

        for index, row in sequences_with_functions.iterrows():
            sequence_type = row['type']
            if "unknown_orfs" in sequence_type:
                sequences_with_functions.at[index, 'local'] = "ToBePredicted"
            if "insilico" in sequence_type:
                sequences_with_functions.at[index, 'local'] = "Random"
        sequences_with_functions.loc[:, 'local'] = sequences_with_functions['local'].fillna("Missing")
        sequences_with_functions = sequences_with_functions.drop_duplicates(subset='orf_id', keep='first') #first will keep positive_orfs
        sequences_with_functions = sequences_with_functions.drop_duplicates(subset='aa_seq', keep='first') #first will keep positive_orfs

            
        sequences_with_functions.to_csv(self.sequencesWithFunctions, index=False)

