import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import re
from protlearn.preprocessing import remove_unnatural

from ..pipeline import PipelineStructure

class GTFtoSeq(PipelineStructure):

    def __init__(self, gtf_file=None, fasta_file=None, utr_length=25):
        self.gtf =  pd.read_csv(gtf_file, sep='\t', header=None)
        self.gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        self.fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
        self.utr_length = utr_length

    def extract_sequences(self):
        """
        Extracts sequences from the GTF file and returns a DataFrame containing the CDS and transcript data.

        Every ORF is handled the same way, whatever order the GTF lists its CDS rows in: CDS exons are ordered by
        genomic position and joined 5' to 3' on the ORF's strand, and the flanks are taken just outside the CDS.

        Returns:
            pandas.DataFrame: A DataFrame containing the following columns:
                - orf_id: The identifier for the open reading frame (ORF).
                - cds_seq: The coding sequence (CDS) sequence, start codon to last sense codon.
                - cds_chr: The chromosome where the CDS is located.
                - cds_starts: The CDS coordinate at the start codon (the highest coordinate on the minus strand).
                - cds_ends: The CDS coordinate at the 3' end (the lowest coordinate on the minus strand).
                - cds_strand: The strand of the CDS.
                - aa_seq: The amino acid sequence translated from the CDS.
                - transcript_starts: The start position of the transcript.
                - transcript_ends: The end position of the transcript.
                - utr_5: The genomic sequence immediately 5' of the start codon.
                - utr_3: The genomic sequence immediately 3' of the stop codon.
        """

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
        cds_rows = pd.DataFrame({"orf_id": cds_ids, "cds_seq": cds_seqs, "cds_chr": cds_chr, "cds_strand": cds_strands, "cds_starts": cds_starts, "cds_ends": cds_ends})

        # Order each ORF's CDS exons 5' to 3' on its strand by genomic position (not by GTF row order) and join them;
        # cds_lo/cds_hi are the genomic CDS bounds used for the flanks
        cds_rows['exon_order'] = np.where(cds_rows['cds_strand'] == '+', cds_rows['cds_starts'], -cds_rows['cds_starts'])
        cds_rows = cds_rows.sort_values(['orf_id', 'exon_order'], kind='stable')
        cds_df = cds_rows.groupby('orf_id').agg(cds_seq=('cds_seq', ''.join),
                                                cds_chr=('cds_chr', 'first'),
                                                cds_strand=('cds_strand', 'first'),
                                                cds_lo=('cds_starts', 'min'),
                                                cds_hi=('cds_ends', 'max')).reset_index()
        minus = cds_df['cds_strand'] == '-'
        cds_df['cds_starts'] = cds_df['cds_lo'].where(~minus, cds_df['cds_hi'])
        cds_df['cds_ends'] = cds_df['cds_hi'].where(~minus, cds_df['cds_lo'])
        cds_df = pd.concat([cds_df[~minus], cds_df[minus]], ignore_index=True)
        cds_df = cds_df[['orf_id', 'cds_seq', 'cds_chr', 'cds_starts', 'cds_ends', 'cds_strand', 'cds_lo', 'cds_hi']]

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
        
        cds_df = cds_df.rename(columns={'cds_protein_seq': 'aa_seq'})

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
        cds_and_transcript['cds_lo'] = cds_and_transcript['cds_lo'].astype(int)
        cds_and_transcript['cds_hi'] = cds_and_transcript['cds_hi'].astype(int)

        # Add utr_5 and utr_3: the genomic sequence just outside the CDS on the ORF's strand (the 3 nt after the CDS are the stop codon)
        cds_and_transcript['utr_5'] = ""
        cds_and_transcript['utr_3'] = ""

        for row in range(len(cds_and_transcript)):
            r = cds_and_transcript.iloc[row]
            if r['cds_strand'] == '+':
                utr_5 = dna_converter(r['cds_chr'], r['transcript_starts'], r['cds_lo']-1, '+', self.fasta_dict)
                utr_3 = dna_converter(r['cds_chr'], r['cds_hi']+4, r['transcript_ends'], '+', self.fasta_dict)
            else:
                utr_5 = dna_converter(r['cds_chr'], r['cds_hi']+1, r['transcript_ends'], '-', self.fasta_dict)
                utr_3 = dna_converter(r['cds_chr'], r['transcript_starts'], r['cds_lo']-4, '-', self.fasta_dict)
            cds_and_transcript.iloc[row, cds_and_transcript.columns.get_loc('utr_5')] = utr_5
            cds_and_transcript.iloc[row, cds_and_transcript.columns.get_loc('utr_3')] = utr_3

        # Ensure 5' and 3' upstream regions are utr_lengths long and add Xs if it is not
        utr_length = self.utr_length 
        utr_length = int(utr_length)
        cds_and_transcript['utr_5'] = cds_and_transcript['utr_5'].str.pad(width= utr_length, side='right', fillchar='X')
        cds_and_transcript['utr_3'] = cds_and_transcript['utr_3'].str.pad(width= utr_length, side='left', fillchar='X')
        cds_and_transcript['utr_5'] = cds_and_transcript['utr_5'].str[- utr_length:]
        cds_and_transcript['utr_3'] = cds_and_transcript['utr_3'].str[:utr_length]
        cds_and_transcript = cds_and_transcript.drop(columns=['cds_lo', 'cds_hi'])

        return cds_and_transcript