# Checkpoints directory

This directory is reserved for **released model checkpoints and local model assets** used by the repository.

The GitHub repository does **not** store the full released checkpoints used in the study.  
Instead:

- **GitHub** stores code, documentation, and demo assets.
- **Zenodo** stores the larger reproducibility assets, including trained checkpoints.
- Users should download the required model files separately and place them under the expected repository-relative paths.

---

## What should be placed here

After downloading the released model files from Zenodo, place them under paths such as:

```text
checkpoints/
├── MOFLLaMA/
├── pretrain_result.pt
├── finetune_result.pt
└── instructor_xl/
```

## Notes

- `MOFLLaMA/` is the released MOFLLaMA checkpoint directory.
- `pretrain_result.pt` is the stage-I pretrained MOFMeld bridge checkpoint.
- `finetune_result.pt` is the stage-II fine-tuned MOFMeld bridge checkpoint.
- `instructor_xl/` is the retrieval embedding model directory used for MOFLLaMA-KG retrieval.