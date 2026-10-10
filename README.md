<p align="center">
  <img src="https://github.com/user-attachments/assets/356d7ec1-0035-48b8-a4d7-c544cade92ca" alt="ShortStop logo" width="500"/>
</p>

## What is ShortStop?

**ShortStop** is a machine learning framework for identifying smORF-encoded microproteins (<150aa) that share biochemical similarity to highly annotated microproteins. It distinguishes between:

- **SAMs** (Swiss-Prot Analog Microproteins): smORFs that resemble microproteins in Swiss-Prot, the reviewed  section of the UniProt Knowledgebase.
- **PRISMs** (Physicochemically Resembling In Silico Microproteins): smORFs that resemble synthetic sequences.

ShortStop also supports the generation of matched negative control microprotein sequences (i.e., PRISMS) for downstream applications.

---

## Why Use ShortStop?

Thousands of smORFs are actively translated in the human transcriptome, yet the functions of their encoded microproteins remain unknown. 

Protein function is conventionally inferred from evidence of evolutionary selection. However, most encoded microproteins are evolutionarily young and lack detectable sequence conservation across species. 

ShortStop provides a homology-independent approach to identify microproteins that share key physicochemical properties with known proteins, even in the absence of sequence similarity. 

ShortStop classifications supports hypothesis-driven studies to "deorphanize" the function of translating smORFs. It also provides a foundation for future benchmarking of new algorithms and methods as data emerges from ongoing studies.

---

## Key Features

- Classifies smORFs as SAMs or PRISMs using a pre-trained ML model
- Generates composition-matched negative control microprotein sequences
- Extracts physicochemical features from input smORFs
- Supports model training, prediction, and feature extraction as CLI modules

---

## Requirements

You’ll need:

1. A GTF file of smORFs that must contain CDS and transcripts features
2. A matched reference genome (e.g., hg38, which automatically downloads upon initiating demo mode).

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

### Option 2 – Clone and Install Locally
```bash
git clone https://github.com/brendan-miller-salk/ShortStop.git
cd ShortStop
pip install .
```

### ⚠️ If you get a C compilation error during install...
Install a C compiler for your system:

- **Ubuntu/Debian**
  ```bash
  sudo apt-get install build-essential
  ```

- **Fedora/CentOS**
  ```bash
  sudo dnf install gcc
  ```

- **Arch Linux**
  ```bash
  sudo pacman -S base-devel
  ```

- **Windows**  
  Download and install: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

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

Estimated run time is 5 minutes.

---

## Usage

### Predict Mode

```bash
shortstop predict -h
```

Classifies smORFs as `SAM:intracellular`, `SAM:secreted`, or `PRISM`

Prediction takes approximately 5-10 minutes per 10,000 smORF sequences. It is recommended to parallelize (e.g., on a cluster) if attempting to classify >1 million sequences.

#### SAM-Consensus (v1.1)

Training ShortStop involves random sampling: which in silico PRISMs are generated, and how the data are split for training and testing. Retraining therefore gives a slightly different SAM list each time. To show how robust each SAM call is, predict mode also scores every smORF with **100 independently trained ShortStop models**. The standard model's classification is unchanged, and the standard model is one of the 100 (`model_001`).

Each consensus model votes SAM when P(SAM) = P(intracellular) + P(secreted) > 0.5. Every smORF gets a **SAM-Consensus percentage** (the share of the 100 models voting SAM) and four yes/no labels:

| Label | Rule | GENCODE smORFs (n = 7,155) |
|---|---|---|
| **SAM** | standard model classification (unchanged from v1.0) | 606 |
| **SAM-Consensus-50** | > 50% of the 100 models vote SAM | 539 |
| **SAM-Consensus-90** | ≥ 90% of models | 291 |
| **SAM-Consensus-100** | all 100 models | 137 |

The SAM-Consensus labels are nested: every SAM-Consensus-100 smORF is also -90 and -50. Use SAM-Consensus-90 or -100 when you need the most robust set, such as when choosing candidates for experiments. 453 of the 606 standard SAMs are also SAM-Consensus-50; the other 153 depend on the particular training sample.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/consensus_levels_dark.png">
  <img alt="Histogram of SAM-Consensus percentage per smORF with the 50, 90 and 100 levels marked" src="docs/figures/consensus_levels_light.png" width="700">
</picture>

The consensus models (~36 MB) are downloaded and checksum-verified on the first run:

```bash
shortstop predict ... --no_consensus           # standard model only (v1.0 behavior)
shortstop predict ... --consensus_dir /path    # use or download the bundle here (offline / shared installs)
```

**How the 100 models were chosen.** ShortStop was retrained 1,000 times, each time on an independent random PRISM sample and train/test split. The consensus models are the standard model plus the 99 next-highest replicates by XGBoost macro AUC (AUC 0.977–0.986, each measured on that run's own held-out test split).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/consensus_auc_dark.png">
  <img alt="Histogram of XGBoost macro AUC across replicate training runs, with the 100 consensus models and the standard model marked" src="docs/figures/consensus_auc_light.png" width="700">
</picture>

**Why 100 models.** Take two sets of 100 models built from completely separate training runs: their SAM-Consensus-50 lists overlap 92% (Jaccard) and their SAM-Consensus-90 lists 90%. More models add little.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/consensus_elbow_dark.png">
  <img alt="SAM-Consensus list reproducibility between two independent model sets versus the number of models, flattening past about 100 models" src="docs/figures/consensus_elbow_light.png" width="700">
</picture>

**Agreement between individual models.** On GENCODE smORFs, each model votes SAM for a median of 590 smORFs (range 464–750). Any two models' SAM lists overlap by Jaccard 0.582 on average, and Fleiss' κ is 0.711 ("substantial"). The models learn a consistent signal, but a single model's SAM list is only partly reproducible, which is why the SAM-Consensus labels are reported.

Agreement measures robustness to the training sample, not correctness: all 100 models share the same positive training set and features, so they can be confidently wrong together. Full results are in [docs/consensus_validation.md](docs/consensus_validation.md).

---

### In Silico Mode

```bash
shortstop insilico -h
```

Generates in silico microprotein sequences matched to your input smORFs by length, amino acid, and nucleic acid composition

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

Train a custom classifier from your own positive/negative examples.

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
│   ├── sams.csv
│   ├── shortstop_classifications.csv
│   └── consensus/
│       ├── shortstop_consensus.csv   # one row per smORF: SAM + SAM-Consensus-50/90/100
│       └── consensus_summary.txt     # label counts and model agreement for this run
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

For commercial licensing inquiries, contact [brmiller@salk.edu](mailto:brmiller@salk.edu). By downloading or using this tool, you agree to the terms in LICENSE.md and CLA.md.
