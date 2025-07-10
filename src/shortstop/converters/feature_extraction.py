import os
from protlearn.features import ctdd, apaac, cksaap
import pandas as pd
import numpy as np
import time
from collections import Counter
import math


class FeatureExtraction:
    def __init__(self, ids, types, labels, aa_seqs, cds_seqs, upstream_seqs, downstream_seqs, utr_length, k):
        
        """
        Initialize the FeatureExtraction object.

        Args:
            ids (list): List of IDs.
            types (list): List of types.
            labels (list): List of labels.
            aa_seqs (list): List of amino acid sequences.
            cds_seqs (list): List of CDS sequences.
            upstream_seqs (list): List of upstream sequences.
            downstream_seqs (list): List of downstream sequences.
            utr_length (int): Length of the UTR region.
            k (int): Value of k.
        """
        
        self.ids = ids
        self.types = types
        self.labels = labels
        self.aa_seq = aa_seqs
        self.cds_seq = cds_seqs
        self.upstream_seq = upstream_seqs
        self.downstream_seq = downstream_seqs
        self.utr_length = utr_length
        self.k = k

        # Convert the sequences to uppercase 
        for i in range(len(self.aa_seq)):
            self.aa_seq[i] = self.aa_seq[i].upper()

        for i in range(len(self.cds_seq)):
            self.cds_seq[i] = self.cds_seq[i].upper()

        for i in range(len(self.upstream_seq)):
            self.upstream_seq[i] = self.upstream_seq[i].upper()

        for i in range(len(self.downstream_seq)):
            self.downstream_seq[i] = self.downstream_seq[i].upper()

        # If the self.upstream_seq is less than self.utr_length nucleotides, pad with Xs
        for i in range(len(self.upstream_seq)):
            if len(self.upstream_seq[i]) < self.utr_length:
                self.upstream_seq[i] =  self.upstream_seq[i] + 'X' * (self.utr_length - len(self.upstream_seq[i]))

        # Keep only the last self.utr_length nucleotides of upstream_seq
        for i in range(len(self.upstream_seq)):
            self.upstream_seq[i] = self.upstream_seq[i][-self.utr_length:]

        # If the self.downstream_seq is less than self.utr_length nucleotides, pad with Xs
        for i in range(len(self.downstream_seq)):
            if len(self.downstream_seq[i]) < self.utr_length:
                self.downstream_seq[i] ='X' * (self.utr_length - len(self.downstream_seq[i])) + self.downstream_seq[i]

        # Keep only the last self.utr_length nucleotides of downstream_seq
        for i in range(len(self.downstream_seq)):
            self.downstream_seq[i] = self.downstream_seq[i][:self.utr_length]


    def kmer_frequency_list(self, sequences):
        
        """
        Calculate the k-mer frequency list for a given list of sequences.

        Args:
            sequences (list): A list of DNA sequences.

        Returns:
            list: A list of dictionaries, where each dictionary represents the k-mer frequencies for a sequence.
                  The keys of the dictionary are the k-mers, and the values are the corresponding frequencies.
        """
        scaled_freqs_list = []

        for sequence in sequences:
            kmers = [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]
            freqs = Counter(kmers)
            
            # Normalize frequencies within the sequence
            total_kmers = len(kmers)
            freqs = {kmer: freq / total_kmers for kmer, freq in freqs.items()}
            
            # Apply min-max scaling within each sequence
            if freqs:
                min_freq = min(freqs.values())
                max_freq = max(freqs.values())
                range_freq = max_freq - min_freq
                if range_freq > 0:
                    scaled_freqs = {kmer: (freq - min_freq) / range_freq for kmer, freq in freqs.items()}
                else:
                    # In case all k-mers have the same frequency, assign them all a value of 1
                    scaled_freqs = {kmer: 1 for kmer in freqs}
            else:
                scaled_freqs = {}
            
            scaled_freqs_list.append(scaled_freqs)
                    
        return scaled_freqs_list
    
    def kozak_score(self, sequences):
        
        score_list = []
        for sequence in sequences:
                score = 3 * (sequence[0] == 'G') + (sequence[1] == 'C') + (sequence[2] == 'C') + \
                3 * (sequence[3] in ['A', 'G']) + (sequence[4] == 'C') + (sequence[5] == 'C') + \
                3 * (sequence[9] == 'G')
                score_list.append(score)
        return score_list
    
    def feature_extraction(self):
        """
        Extracts features from the given amino acid sequences and returns a DataFrame.

        Returns:
            df (pandas.DataFrame): DataFrame containing the extracted features and corresponding labels and types.
        """
        time_start = time.time()

        features_list = []
        labels_list = []

        for method in [ctdd, cksaap, apaac]:
            
            if method == apaac:
                features, labels = method(self.aa_seq, lambda_=8)
                features = features.astype(np.float64)
            elif method == cksaap:
                features, labels = method(self.aa_seq)
                # Add prefix ctd to the labels
                labels = [f'cksaap{label}' for label in labels]
                # Convert features to float64
                features = features.astype(np.float64)
            else:
                features, labels = method(self.aa_seq)
                # Convert features to float64
                features = features.astype(np.float64)
            
            labels = [str(i) for i in labels]
            features_list.append(features)
            labels_list += labels

        features = np.concatenate(features_list, axis=1)
        aa_df = pd.DataFrame(features, columns=labels_list)
        aa_df['orf_id'] = self.ids
        
        upstream_sequences = [seq.upper() for seq in self.upstream_seq]
        upstream_freq = self.kmer_frequency_list(upstream_sequences)
        cds_upstream_kmer = pd.DataFrame(upstream_freq)
        cds_upstream_kmer = cds_upstream_kmer.add_prefix('5_prime_')
        cds_upstream_kmer['orf_id'] = self.ids
        
        downstream_sequences = [seq.upper() for seq in self.downstream_seq]
        downstream_freq = self.kmer_frequency_list(downstream_sequences)
        cds_downstream_kmer = pd.DataFrame(downstream_freq)
        cds_downstream_kmer = cds_downstream_kmer.add_prefix('3_prime_')
        cds_downstream_kmer['orf_id'] = self.ids
        
        kmer_df = pd.merge(cds_upstream_kmer, cds_downstream_kmer, on='orf_id')
        kmer_df = kmer_df.fillna(0)
        
        kozak_context = [seq[-6:] + seq2[:4] for seq, seq2 in zip(self.upstream_seq, self.cds_seq)]
        kozak_score = self.kozak_score(kozak_context)
        kozak_df = pd.DataFrame(kozak_score, columns=['kozak_score'])
        kozak_df['orf_id'] = self.ids

        first_50 = [seq[:50] for seq in self.cds_seq]
        first_50_kmer = self.kmer_frequency_list(first_50)
        first_50_kmer = pd.DataFrame(first_50_kmer)
        first_50_kmer = first_50_kmer.add_prefix('first_50_')
        first_50_kmer['orf_id'] = self.ids
        first_50_kmer = first_50_kmer.fillna(0)

        # Merge all features
        df = pd.merge(kmer_df, kozak_df, on='orf_id')
        df = pd.merge(df, first_50_kmer, on='orf_id')
        df = pd.merge(df, aa_df, on='orf_id')
                
        # Add labels and types
        df['label'] = self.labels
        df['type'] = self.types

        return df