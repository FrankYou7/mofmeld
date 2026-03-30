# MOFMeld

MOFMeld is a MOF-oriented language and multimodal modeling project for carbon-capture-related reasoning and property prediction.

This repository contains the **released codebase**, documentation, and **runnable demo inference workflows** for two complementary components:

- **MOFLLaMA**: a literature-specialized large language model fine-tuned for MOF question answering, benchmark evaluation, and retrieval-assisted inference.
- **MOFMeld**: a structure-language fusion framework that integrates CHGNet-derived structure embeddings with a frozen MOFLLaMA backbone through a bridge module for structure-aware reasoning and MOF property prediction.

Large datasets, trained checkpoints, retrieval stores, and archived result files are hosted separately on **Zenodo**.

---

## Repository overview

This repository follows the following release policy:

- **Training code**: fully released to preserve the original experimental workflow, but **not packaged as lightweight demo training**.
- **Inference code**: two runnable demos are provided:
  - one for **MOFLLaMA + KG-assisted retrieval**
  - one for **MOFMeld structure-aware property prediction**
- **GitHub** stores:
  - source code
  - documentation
  - small demo assets
  - runnable demo scripts
- **Zenodo** stores:
  - processed datasets
  - trained checkpoints
  - retrieval indices
  - structure embeddings
  - archived prediction/evaluation files

---

## Project structure

```text
mofmeld/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── docs/
├── configs/
├── src/
│   ├── common/
│   ├── mofllama/
│   ├── mofmeld/
│   └── baselines/
├── scripts/
│   ├── train_mofllama.sh
│   ├── run_mofllama_demo.sh
│   ├── extract_chgnet_embeddings.sh
│   ├── pretrain_mofmeld_bridge.sh
│   ├── finetune_mofmeld_bridge.sh
│   ├── run_mofmeld_demo.sh
│   └── slurm/
├── data_demo/
│   ├── mofllama/
│   └── mofmeld/
├── results_demo/
├── checkpoints/
└── data/
```

## Components

### 1. MOFLLaMA

MOFLLaMA is a MOF-specialized LLM trained on literature-derived MOF QA data.

It supports:

- supervised fine-tuning on MOF QA data
- evaluation on held-out QA and MCQ benchmarks
- retrieval-assisted inference using MOFLLaMA-KG resources

#### Main code

`src/mofllama/training/train_mofllama.py`  
Supervised fine-tuning script for MOFLLaMA.

`src/mofllama/inference/run_kg_grounded_inference_demo.py`  
Interactive demo for MOFLLaMA + KG-grounded retrieval and answering.

`src/mofllama/retrieval/build_faiss_index.py`  
Utility for building the retrieval index used in MOFLLaMA-KG.

`src/mofllama/kg/build_citation_metadata.py`  
Utility for preparing citation metadata for retrieved references.

#### Main scripts

`scripts/train_mofllama.sh`  
Full MOFLLaMA training workflow.

`scripts/slurm/train_mofllama.slurm`  
Example Slurm submission script for HPC training.

`scripts/run_mofllama_demo.sh`  
Interactive demo launcher for MOFLLaMA + KG retrieval.

---

### 2. MOFMeld

MOFMeld is a structure-language fusion framework that combines:

- a frozen MOFLLaMA backbone
- CHGNet-derived 64-dimensional structure embeddings
- a bridge module trained in two stages  
  - stage-I pretraining  
  - stage-II fine-tuning

It supports:

- structure-text pretraining
- structure-conditioned question answering
- property prediction for MOF structural and adsorption properties

#### Main code

`src/mofmeld/models/mof_bridge.py`  
Unified MOFMeld bridge / multimodal model definition.

`src/mofmeld/embeddings/extract_chgnet_embeddings.py`  
Extracts 64-dimensional structure embeddings from CIF files using CHGNet.

`src/mofmeld/data/build_pretrain_tasks.py`  
Builds stage-I pretraining task files.

`src/mofmeld/data/pretrain_dataset.py`  
Dataset utilities for stage-I pretraining.

`src/mofmeld/training/pretrain_bridge.py`  
Stage-I pretraining script for the bridge module.

`src/mofmeld/data/build_finetune_tasks.py`  
Builds stage-II QA-style fine-tuning data.

`src/mofmeld/data/finetune_dataset.py`  
Dataset utilities for stage-II fine-tuning.

`src/mofmeld/training/finetune_bridge_ddp.py`  
Stage-II distributed fine-tuning script.

`src/mofmeld/inference/run_property_prediction_demo.py`  
Interactive MOFMeld demo for structure-aware property prediction.

`src/mofmeld/evaluation/evaluate_property_predictions.py`  
Evaluation utilities for prediction outputs.

#### Main scripts

`scripts/extract_chgnet_embeddings.sh`  
Extracts CHGNet-based structure embeddings from CIF files.

`scripts/pretrain_mofmeld_bridge.sh`  
Full stage-I bridge pretraining workflow.

`scripts/finetune_mofmeld_bridge.sh`  
Full stage-II bridge fine-tuning workflow.

`scripts/run_mofmeld_demo.sh`  
Interactive MOFMeld demo launcher.

---

### 3. Baselines

This repository also includes baseline support code for CHGNet-based property prediction.

#### Main code

`src/baselines/chgnet/train_chgnet_baseline.py`  
`src/baselines/chgnet/run_chgnet_inference.py`  
`src/baselines/chgnet/evaluate_chgnet_predictions.py`

These scripts correspond to the baseline checkpoints and prediction outputs archived on Zenodo.

---

# Demo workflows

This repository provides two runnable demo inference workflows.

---

## A. MOFLLaMA + KG demo

This demo allows the user to enter a natural-language question in the terminal.

The system retrieves relevant context from a small demo FAISS store, generates an answer using MOFLLaMA, and prints the corresponding literature references.

Run:

```bash
bash scripts/run_mofllama_demo.sh
```

### Required demo assets

GitHub includes small demo retrieval assets under:

```
data_demo/mofllama/retrieval_store/faiss_demo/
data_demo/mofllama/metadata/citations_demo.csv
```

You must also place the required checkpoints under:

```
checkpoints/mofllama_sft/
checkpoints/instructor_xl/
```

---

## B. MOFMeld demo

This demo allows the user to:

- enter the name of a demo MOF structure (for example `hmof-9`)
- enter a natural-language property question
- obtain a generated answer and the inference time for that single sample.

Run:

```bash
bash scripts/run_mofmeld_demo.sh
```

### Required demo assets

GitHub includes small demo structure files under:

```
data_demo/mofmeld/sample_cifs/
data_demo/mofmeld/sample_embeddings/
```

The CIF filename and embedding filename must share the same stem, for example:

```
hmof-9.cif
hmof-9.pt
```

You must also place the required checkpoints under:

```
checkpoints/mofllama_sft/
checkpoints/mofmeld_bridge_finetuned.pt
```

---

# Training workflows

Training code is fully released to document the original workflow used in the study.

### MOFLLaMA training

```bash
bash scripts/train_mofllama.sh
```

### MOFMeld stage-I pretraining

```bash
bash scripts/pretrain_mofmeld_bridge.sh
```

### MOFMeld stage-II fine-tuning

```bash
bash scripts/finetune_mofmeld_bridge.sh
```

### CHGNet baseline preprocessing / training

See the scripts under:

```
src/baselines/chgnet/
src/mofmeld/embeddings/
```

These scripts are intended for full reproduction rather than lightweight demo execution.

---

# Data and checkpoints

## GitHub

This repository contains:

- code  
- documentation  
- small demo assets  
- demo launch scripts  

## Zenodo

Zenodo contains the larger reproducibility assets:

- processed datasets  
- trained checkpoints  
- retrieval stores  
- structure embeddings  
- archived prediction outputs  
- external validation files  

---

# Expected local layout after downloading Zenodo assets

Place downloaded assets under the following repository-relative paths.

### Checkpoints

```
checkpoints/
├── mofllama_sft/
├── mofmeld_bridge_pretrained.pt
├── mofmeld_bridge_finetuned.pt
└── instructor_xl/
```

### Data

```
data/
├── mofllama/
│   ├── train_dataset_llama3_format_only.jsonl
│   └── ...
├── mofmeld_pretrain/
│   ├── prediction.jsonl
│   ├── correlation.jsonl
│   └── association.jsonl
├── mofmeld/
│   └── finetune_qa.jsonl
└── embeddings/
    ├── qmof_embeddings/
    └── qmof_hmof_embeddings/
```

See:

```
checkpoints/README.md
data/README.md
```

for more detailed placement instructions.

---

# Environment

This project depends on:

- Python
- PyTorch
- Transformers
- CHGNet
- pymatgen
- sentence-transformers
- FAISS

Install dependencies using:

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
```

Depending on your environment and hardware, additional CUDA-compatible packages may be required.

---

# Notes on released assets

This repository is intended to preserve the original workflow used in the study while replacing machine-specific absolute paths with portable repository-relative paths.

- Original publisher-provided full-text articles are not redistributed here.
- Large checkpoints, retrieval indices, and processed datasets are stored on Zenodo rather than GitHub.
- Demo-scale assets are included only where needed for runnable inference examples.

---

# Citation

If you use this repository or the archived assets, please cite the associated manuscript and Zenodo release.

(Replace this section with the exact manuscript citation and Zenodo DOI.)

---

# License

Please see the `LICENSE` file for code usage terms.

Dataset and model redistribution may be subject to the licenses and terms associated with the original sources and base models.
