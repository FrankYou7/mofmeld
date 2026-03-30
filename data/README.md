# Data directory

This directory is reserved for **released datasets, processed resources, and local preprocessing outputs** used by the repository.

The GitHub repository does **not** store the full training/evaluation datasets used in the study.  
Instead:

- **GitHub** provides code, documentation, and small demo assets.
- **Zenodo** provides the larger reproducibility assets, including processed datasets, retrieval resources, checkpoints, and archived outputs.
- Users should download the required data files separately and place them under the expected repository-relative paths.

---

## What should be placed here

Depending on the workflow you want to reproduce, place the downloaded Zenodo data and locally generated preprocessing outputs under paths such as the following.

```text
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
├── coremof/
│   ├── coremof_4props.csv
│   └── coremof_co2max_top50.csv
├── embeddings/
│   ├── qmof_embeddings/
│   ├── qmof_hmof_embeddings/
│   └── train_all_embeddings/
├── cifs/
│   └── train_cif/
└── jsons/
```

## Notes

- `chgnet_hmof/` stores the CHGNet baseline training data.
- `mofllama/` stores MOFLLaMA training, evaluation, and KG retrieval resources.
- `mofmeld_pretrain/` stores the stage-I MOFMeld pretraining task files.
- `mofmeld_finetune/` stores the stage-II MOFMeld fine-tuning task file.
- `mofmeld_metadata/` stores released train/test MOF ID lists.
- `coremof/` stores released CoRE-MOF evaluation outputs.
- `embeddings/`, `cifs/`, and `jsons/` may contain locally generated preprocessing inputs or outputs used by embedding extraction and multimodal training workflows.

---

## Optional evaluation subset files

If you want to evaluate predictions on your own subset, you may additionally prepare a user-defined subset file, for example:

```text
data/mofmeld_metadata/your_subset_list.txt
```

or

```text
data/mofmeld_metadata/your_subset_list.csv
```

where:

- `.txt` contains one MOF name per line
- `.csv` contains a column named `mof_name`