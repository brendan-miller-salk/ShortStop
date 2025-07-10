import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import re
from protlearn.preprocessing import remove_unnatural

from ..pipeline import PipelineStructure

class GTFtoSeq(PipelineStructure):

    def __init__(self, gtf_file=None, fasta_file=None, utr_length=25, cds_order = 'First'):
        self.gtf =  pd.read_csv(gtf_file, sep='\t', header=None)
        self.gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
        self.cds_order = cds_order
        self.utr_length = utr_length

    def extract_sequences(self):
        """
        Extracts sequences from the GTF file and returns a DataFrame containing the CDS and transcript data.

        Returns:
            pandas.DataFrame: A DataFrame containing the following columns:
                - orf_id: The identifier for the open reading frame (ORF).
                - cds_seq: The coding sequence (CDS) sequence.
                - cds_chr: The chromosome where the CDS is located.
                - cds_starts: The start position of the CDS.
                - cds_ends: The end position of the CDS.
                - cds_strand: The strand of the CDS.
                - aa_seq: The amino acid sequence translated from the CDS.
                - transcript_starts: The start position of the transcript.
                - transcript_ends: The end position of the transcript.
                - utr_5: The 5' upstream region of the CDS.
                - utr_3: The 3' upstream region of the CDS.
        """
    def extract_sequences(self):
        
        def dna_converter(seqname, start, end, strand, fasta_dict):
            sequence = fasta_dict[seqname].seq
            if strand == "+":
                return str(sequence[start-1:end]).replace(" ", "")
            else:
                return str(sequence[start-1:end].reverse_complement()).replace(" ", "")

        # Prepare transcript data
        transcript_ids = []
        transcript_chr = []
        transcript_starts = []
        transcript_ends = []

        if self.cds_order == "First":
            print("🔔: Don't forget double check the CDS order of your GTF file. This is important for concatenating the CDS sequences, specifically on the reverse strand. Failure to do so could result in incorrect CDS sequences.")

        cds_ids = []
        cds_seqs = []
        cds_chr = []
        cds_starts = []
        cds_ends = []
        cds_strands = []

        for _, row in self.gtf.iterrows():
            if row["feature"] == "transcript":
                transcript_id = re.findall('gene_id (.+?);', row["attribute"])[0]
                transcript_ids.append(transcript_id)
                transcript_chr.append(row["seqname"])
                transcript_starts.append(row['start'])
                transcript_ends.append(row['end'])

            elif row["feature"] == "CDS":
                cds_id = re.findall('gene_id (.+?);', row["attribute"])[0]
                cds_ids.append(cds_id)
                cds_seqs.append(dna_converter(row["seqname"], row["start"], row["end"], row["strand"], self.fasta_dict))
                cds_chr.append(row["seqname"])
                cds_starts.append(row['start'])
                cds_ends.append(row['end'])
                cds_strands.append(row['strand'])

        transcript_df = pd.DataFrame({"orf_id": transcript_ids ,"transcript_starts": transcript_starts, "transcript_ends": transcript_ends})
        cds_df = pd.DataFrame({"orf_id": cds_ids, "cds_seq": cds_seqs, "cds_chr": cds_chr, "cds_strand": cds_strands, "cds_starts": cds_starts, "cds_ends": cds_ends})
        csd_df_forward = cds_df[cds_df['cds_strand'] == "+"]

        if self.cds_order == "Last":
            csd_df_forward = csd_df_forward[csd_df_forward['cds_strand'] == "+"].groupby('orf_id').agg({'cds_seq': lambda x: ''.join(x),
                                        'cds_chr': 'first',
                                        'cds_starts': 'first', 
                                        'cds_ends': 'last',
                                        'cds_strand': 'first'}).reset_index() 
            
            cds_df_reverse = cds_df[cds_df['cds_strand'] == "-"]
            cds_df_reverse = cds_df_reverse[cds_df_reverse['cds_strand'] == "-"].groupby('orf_id').agg({'cds_seq': lambda x: ''.join(x[::-1]),
                                        'cds_chr': 'first',
                                        'cds_starts': 'first', 
                                        'cds_ends': 'last',
                                        'cds_strand': 'first'}).reset_index()
            # Switch cds_starts and cds_ends
            cds_df_reverse['cds_starts'], cds_df_reverse['cds_ends'] = cds_df_reverse['cds_ends'], cds_df_reverse['cds_starts']
        else:
            csd_df_forward = csd_df_forward[csd_df_forward['cds_strand'] == "+"].groupby('orf_id').agg({'cds_seq': lambda x: ''.join(x),
                                    'cds_chr': 'first',
                                    'cds_starts': 'first', 
                                    'cds_ends': 'last',
                                    'cds_strand': 'first'}).reset_index() 
        
            cds_df_reverse = cds_df[cds_df['cds_strand'] == "-"]
            cds_df_reverse = cds_df_reverse[cds_df_reverse['cds_strand'] == "-"].groupby('orf_id').agg({'cds_seq': lambda x: ''.join(x),
                                        'cds_chr': 'first',
                                        'cds_starts': 'first', 
                                        'cds_ends': 'last',
                                        'cds_strand': 'last'}).reset_index()

        cds_df = pd.concat([csd_df_forward, cds_df_reverse], ignore_index=True)
    
        cds_df['cds_protein_seq'] = cds_df['cds_seq'].apply(lambda x: Seq(x).translate()) # Transale cds_seq into protein sequence
        cds_df['cds_protein_seq'] = cds_df['cds_protein_seq'].astype(str)
        cds_df['cds_protein_seq'] = cds_df['cds_protein_seq'].str.replace('*', '', regex=False)
        
        # Only keep the sequence if it starts with M
        cds_df = cds_df[cds_df['cds_protein_seq'].str.startswith('M')]
        
        # Only keep the sequence if it is over 30 amino acids long and under 150 amino acids long
        cds_df = cds_df[cds_df['cds_protein_seq'].str.len() >= 9]
        cds_df = cds_df[cds_df['cds_protein_seq'].str.len() <= 150]
        
        # Only keep the sequence if it contains natural amino acids
        sequences = cds_df['cds_protein_seq'].tolist()

        # Clean the sequences
        cleaned_sequences = remove_unnatural(sequences)
        
        # Update the 'cds_protein_seq' column with the cleaned sequences
        cds_df = cds_df[cds_df['cds_protein_seq'].isin(cleaned_sequences)]
        
        cds_df.columns = ['orf_id', 'cds_seq', 'cds_chr', 'cds_starts', 'cds_ends', 'cds_strand', 'aa_seq'] # Change column names

        cds_and_transcript = pd.merge(cds_df, transcript_df, on='orf_id', how='left') # Merge transcript_df and cds_df
  
        # Check if there are na in the data frame
        if cds_and_transcript.isna().values.any():
            print("🚨 There are missing sequences for the CDS and Transcript data. These genes have been removed, but consider checking the genes.")
            # Remove na
            cds_and_transcript = cds_and_transcript.dropna()

        cds_and_transcript['cds_starts'] = cds_and_transcript['cds_starts'].astype(int) # Make sure the data type is int
        cds_and_transcript['cds_ends'] = cds_and_transcript['cds_ends'].astype(int) # Make sure the data type is int
        cds_and_transcript['transcript_starts'] = cds_and_transcript['transcript_starts'].astype(int) # Make sure the data type is int
        cds_and_transcript['transcript_ends'] = cds_and_transcript['transcript_ends'].astype(int) # Make sure the data type is int

        # Add utr_5 and utr_3
        cds_and_transcript['utr_5'] = ""
        cds_and_transcript['utr_3'] = ""

        for row in range(len(cds_and_transcript)):
            if cds_and_transcript.iloc[row]['cds_strand'] == '+':
                cds_and_transcript.iloc[row, cds_and_transcript.columns.get_loc('utr_5')] = dna_converter(cds_and_transcript.iloc[row]['cds_chr'], cds_and_transcript.iloc[row]['transcript_starts'], cds_and_transcript.iloc[row]['cds_starts']-1, cds_and_transcript.iloc[row]['cds_strand'], self.fasta_dict)
                cds_and_transcript.iloc[row, cds_and_transcript.columns.get_loc('utr_3')] = dna_converter(cds_and_transcript.iloc[row]['cds_chr'], cds_and_transcript.iloc[row]['cds_ends']+4, cds_and_transcript.iloc[row]['transcript_ends'], cds_and_transcript.iloc[row]['cds_strand'], self.fasta_dict)
            else:
                cds_and_transcript.iloc[row, cds_and_transcript.columns.get_loc('utr_5')] = dna_converter(cds_and_transcript.iloc[row]['cds_chr'], cds_and_transcript.iloc[row]['cds_starts']+1, cds_and_transcript.iloc[row]['transcript_ends'], cds_and_transcript.iloc[row]['cds_strand'], self.fasta_dict)
                cds_and_transcript.iloc[row, cds_and_transcript.columns.get_loc('utr_3')] = dna_converter(cds_and_transcript.iloc[row]['cds_chr'], cds_and_transcript.iloc[row]['transcript_starts'], cds_and_transcript.iloc[row]['cds_ends']-4, cds_and_transcript.iloc[row]['cds_strand'], self.fasta_dict)
        
        # Ensure 5' and 3' upstream regions are utr_lengths long and add Xs if it is not
        utr_length = self.utr_length 
        utr_length = int(utr_length)
        cds_and_transcript['utr_5'] = cds_and_transcript['utr_5'].str.pad(width= utr_length, side='right', fillchar='X')
        cds_and_transcript['utr_3'] = cds_and_transcript['utr_3'].str.pad(width= utr_length, side='left', fillchar='X')
        cds_and_transcript['utr_5'] = cds_and_transcript['utr_5'].str[- utr_length:]
        cds_and_transcript['utr_3'] = cds_and_transcript['utr_3'].str[:utr_length]

        return cds_and_transcript