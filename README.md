<p align="center">
  <img src="https://github.com/user-attachments/assets/356d7ec1-0035-48b8-a4d7-c544cade92ca" alt="ShortStop logo" width="500"/>
</p>

## What is ShortStop?

**ShortStop** is a machine learning framework for identifying smORF-encoded microproteins (<150aa) that share biochemical similarity to highly annotated microproteins. It distinguishes between:

- **SAMs** (Swiss-Prot Analog Microproteins): smORFs that resemble microproteins in Swiss-Prot, the reviewed  section of the UniProt Knowledgebase.
- **PRISMs** (Physicochemically Resembling In Silico Microproteins): smORFs that resemble synthetic sequences.

ShortStop also supports the generation of matched negative control microprotein sequences (i.e., PRISMS) for downstream applications.

---

## Why use ShortStop?

Thousands of smORFs are actively translated in the human transcriptome, yet the functions of their encoded microproteins remain unknown. Protein function is conventionally inferred from evidence of evolutionary selection. However, most encoded microproteins are evolutionarily young and lack detectable sequence conservation across species. ShortStop provides a homology-independent approach to identify microproteins that share key physicochemical properties with known proteins, even in the absence of sequence similarity. These classifications may support more hypothesis-driven functional studies.

---

## Key Features

- Classifies smORFs as SAMs or PRISMs using a pre-trained ML model
- Generates composition-matched negative control microprotein sequences
- Extracts physicochemical features from input smORFs
- Supports model training, prediction, and feature extraction as CLI modules

---

## Requirements

You’ll need:

1. A GTF file of smORFs (with CDS, exons, and transcripts)
2. A reference genome (e.g., hg38) used to generate the GTF

---

## Installation

> ✅ Optionally create a conda environment (recommended):
> ```bash
> conda create -n shortstop python=3.9
> conda activate shortstop
> ```

### Option 1 – Direct from GitHub (recommended)
```bash
pip install git+https://github.com/brendan-miller-salk/ShortStop.git
```

### Option 2 – Clone and install locally
```bash
git clone https://github.com/brendan-miller-salk/ShortStop.git
cd ShortStop
pip install .
```

---

## Quickstart (Demo Mode)

Run the built-in demo, which exercises all core modules:

```bash
shortstop demo
```

If reference genome are not provided manually, they will be auto-downloaded into the source code directory. These file in total are ~5 GB.

To manually specify references and avoid downloads:

```bash
shortstop demo \
  --genome demo_data/hg_38_primary.fa \
  --positive_gtf demo_data/gencode.v43.primary_assembly.basic.annotation.gtf
```

---

## Usage

### Predict Mode

```bash
shortstop predict -h
```

Classify smORFs as `SAM:intracellular`, `SAM:secreted`, or `PRISM`.

---

### In Silico Mode

```bash
shortstop insilico -h
```

Generate decoy microprotein sequences matched to your input smORFs.

---

### Feature Extraction

```bash
shortstop feature_extract -h
```

Extract nucleotide and amino acid features from smORFs.

---

### Training Mode

```bash
shortstop train -h
```

Train a custom classifier from your own positive/negative examples. Supports parallelization.

---

## Output Structure

```
shortstop_output/
├── features/
│   └── extracted_features_of_smorfs.csv
├── predictions/
│   ├── sam_secreted.csv
│   ├── sam_intracellular.csv
│   ├── prisms.csv
│   └── shortstop_classifications.csv
├── sequences/
│   ├── positive_unknown_insilico_sequences.csv
│   ├── positive_and_unknown_sequences.csv
│   ├── unknown_sequences.csv
│   └── positive_proteins.gtf
```

---

## Contact

ShortStop is in continuous development. For any questions or suggestions, contact [brmiller@salk.edu](mailto:brmiller@salk.edu) or open an issue on GitHub.

---

## Citation

If you used ShortStop, please cite it in your manuscript:  
*coming soon*

## License and Contributions

This project is licensed for **non-commercial academic research use only**.  
See [LICENSE.md](./LICENSE.md) for full terms.

By contributing to this repository, you agree to the [Contributor License Agreement (CLA)](./CLA.md).  
Please read our [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting code or issues.

For commercial licensing inquiries, contact [brmiller@salk.edu](mailto:brmiller@salk.edu).