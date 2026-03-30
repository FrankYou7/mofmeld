# MOFMeld

MOFMeld is a MOF-oriented language and multimodal modeling project for carbon-capture-related reasoning and property prediction.

This repository contains the **released codebase**, documentation, and **runnable demo inference workflows** for two complementary components:

- **MOFLLaMA**: a literature-specialized large language model fine-tuned for MOF question answering, benchmark evaluation, and retrieval-assisted inference.
- **MOFMeld**: a structure-language fusion framework that integrates CHGNet-derived structure embeddings with a frozen MOFLLaMA backbone through a bridge module for structure-aware reasoning and MOF property prediction.

Large datasets, trained checkpoints, retrieval stores, and archived result files are hosted separately on **Zenodo**.

---

## Repository overview

This repository follows the following release policy:

- **Training code**: fully released to preserve the original experimental workflow
- **Inference code**: two runnable demos are provided  
  - one for **MOFLLaMA + KG-assisted retrieval**  
  - one for **MOFMeld structure-aware property prediction**
- **GitHub** stores  
  - source code  
  - documentation  
  - small demo assets  
  - runnable demo scripts
- **Zenodo** stores  
  - processed datasets  
  - trained checkpoints  
  - retrieval indices  
  - structure embeddings  
  - archived prediction/evaluation files

---

## Project structure

```text
MOFMELD/
├── README.md
├── .gitignore
├── checkpoint/
│   └── readme.md
├── data/
│   └── readme.md
├── data_demo/
│   ├── mofllama/
│   │   ├── metadata/
│   │   │   └── citations_metadata.csv
│   │   └── retrieval_store/
│   │       └── embedding/
│   │           ├── index.faiss
│   │           └── index.pkl
│   └── mofmeld/
│       ├── sample_cifs/
│       └── sample_embeddings/
├── scripts/
│   ├── evaluate_chgnet_prediction.sh
│   ├── extract_chgnet_embeddings.sh
│   ├── finetune_mofmeld_bridge.sh
│   ├── pretrain_mofmeld_bridge.sh
│   ├── run_mofllama_demo.sh
│   ├── run_mofmeld_demo.sh
│   ├── train_chgnet_baseline.sh
│   └── train_mofllama.sh
└── src/
    ├── baselines/
    │   └── chgnet/
    │       ├── evaluate_prediction.py
    │       └── train_chgnet_baseline.py
    ├── mofllama/
    │   ├── inference/
    │   │   └── run_kg_grounded_inference_demo.py
    │   ├── retrieval/
    │   │   └── build_faiss_index.py
    │   └── training/
    │       └── train_mofllama.py
    └── mofmeld/
        ├── data/
        ├── embeddings/
        │   └── extract_chgnet_embeddings.py
        ├── evaluation/
        │   └── evaluate_metrics.py
        ├── inference/
        │   ├── run_property_prediction.py
        │   └── run_property_prediction_demo.py
        ├── models/
        │   └── mof_bridge.py
        └── training/
            ├── finetune_ddp.py
            └── pretrain_bridge.py
```

---

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

#### Main scripts

`scripts/train_mofllama.sh`  
Full MOFLLaMA training workflow.

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

- property prediction for MOF structural and adsorption properties  

#### Main code

`src/mofmeld/models/mof_bridge.py`  
Unified MOFMeld bridge / multimodal model definition.

`src/mofmeld/embeddings/extract_chgnet_embeddings.py`  
Extracts 64-dimensional structure embeddings from CIF files using CHGNet.

`src/mofmeld/training/pretrain_bridge.py`  
Stage-I pretraining script for the bridge module.

`src/mofmeld/training/finetune_ddp.py`  
Stage-II distributed fine-tuning script.

`src/mofmeld/inference/run_property_prediction.py`  
Property prediction script for MOFMeld.

`src/mofmeld/inference/run_property_prediction_demo.py`  
Interactive MOFMeld demo for structure-aware property prediction.

`src/mofmeld/evaluation/evaluate_metrics.py`  
Evaluation script for MAD, R2, and RMSE.

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
Training script for CHGNet baseline models across the six target properties.

`src/baselines/chgnet/evaluate_prediction.py`  
Subset filtering and evaluation script for CHGNet prediction CSV files.

#### Main scripts

`scripts/train_chgnet_baseline.sh`  
Launcher for CHGNet baseline training.

`scripts/evaluate_chgnet_prediction.sh`  
Launcher for CHGNet prediction filtering and evaluation.

---

## Demo workflows

This repository provides two runnable demo inference workflows.

### A. MOFLLaMA + KG demo

Run:

```bash
bash scripts/run_mofllama_demo.sh
```

Required demo assets:

```
data_demo/mofllama/metadata/citations_metadata.csv
data_demo/mofllama/retrieval_store/embedding/
```

Required checkpoints:

```
checkpoint/MOFLLaMA/
checkpoint/instructor_xl/
```

---

### B. MOFMeld demo

Run:

```bash
bash scripts/run_mofmeld_demo.sh
```

Required demo assets:

```
data_demo/mofmeld/sample_cifs/
data_demo/mofmeld/sample_embeddings/
```

Example file pair:

```
hmof-9.cif
hmof-9.pt
```

Required checkpoints:

```
checkpoint/MOFLLaMA/
checkpoint/finetune_result.pt
```

---

## Expected local layout after downloading Zenodo assets

### Checkpoints

```
checkpoint/
├── MOFLLaMA/
├── pretrain_result.pt
├── finetune_result.pt
└── instructor_xl/
```

### Data

```
data/
├── chgnet_hmof/
│   ├── labels.json
│   ├── train/cifs/
│   ├── val/cifs/
│   └── test/cifs/
├── mofllama/
│   ├── mofllama_train_dataset.jsonl
│   ├── qa_test.jsonl
│   ├── easy_mcq.jsonl
│   ├── hard_mcq.jsonl
│   └── kg/
│       ├── mofllama_kg_triples.jsonl
│       └── citations_metadata.csv
├── mofmeld_pretrain/
│   ├── prediction.jsonl
│   ├── correlation.jsonl
│   └── association.jsonl
├── mofmeld_finetune/
│   └── finetune_qa.jsonl
├── mofmeld_metadata/
│   ├── train_hmof_30000.txt
│   └── test_hmof_2769.txt
└── coremof/
    ├── coremof_4props.csv
    └── coremof_co2max_top50.csv
```

See:

```
checkpoint/readme.md
data/readme.md
```

for more detailed placement instructions.
