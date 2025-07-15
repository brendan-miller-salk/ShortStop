#!/bin/bash

cd demo_data

# URLs of the files you need for the demo_data
GTF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.primary_assembly.basic.annotation.gtf.gz"
FA_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/GRCh38.primary_assembly.genome.fa.gz"

# Download the .gtf file
curl -O $GTF_URL

# Decompress the .gtf file
gunzip -f gencode.v43.primary_assembly.basic.annotation.gtf.gz
grep -v '^#' gencode.v43.primary_assembly.basic.annotation.gtf > temp_file.gtf 
mv temp_file.gtf gencode.v43.primary_assembly.basic.annotation.gtf

# Download the .fa file
curl -O $FA_URL

# Decompress the .fa file
gunzip -f GRCh38.primary_assembly.genome.fa.gz
mv GRCh38.primary_assembly.genome.fa hg_38_primary.fa
