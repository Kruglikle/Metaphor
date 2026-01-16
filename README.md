# Metaphor

Small, controlled dataset for studying how contextual embeddings shift between
literal and metaphorical uses of a target word.

## Contents
- `analyze_smallset.py`: analysis script that computes embeddings and reports
  literal vs. metaphorical shifts.
- `small_metaphor_dataset_4words.csv`: default dataset used by the script.
- `dataset.csv`: additional dataset you can experiment with.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python analyze_smallset.py --data_path small_metaphor_dataset_4words.csv --model_name roberta-base --method pca
```

Use `--method umap` to switch the projection method.

## Outputs
The script writes results under `runs/smallset/<timestamp>/`:
- `embeddings.csv`: rows with embedding vectors and PCA coordinates.
- `metrics_by_word.csv`: per-word centroid distances and dispersion.
- `plots/`: per-word shift plots and a global overview.
- `report.md`: markdown report with plots.
- `logs.txt`: run logs and dropped example notes.

## Notes
- Model files are cached under `hf_cache/` and are ignored by Git.
- If a model download fails due to SSL issues, the script retries with SSL
  verification disabled.

## License
MIT
