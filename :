# Multimodal Mastitis Detection – Imaging + Sensor Fusion

This repository contains the code and the Jupyter notebook used to run the multimodal (tabular + thermal imaging) pipeline for early mastitis detection in dairy cows.

The project was originally executed in **Google Colab Pro**. This version has been reorganised so that it can be pushed directly to **GitHub** and later rerun either in Colab or in a local Python environment.

## Repository structure

```text
.
├── notebooks/
│   └── mastitis_multimodal_pipeline_version_submitted.ipynb   # main Colab/Notebook pipeline (as provided)
├── src/
│   ├── __init__.py
│   ├── data_loading.py          # helpers to load/clean the tabular part
│   ├── imaging_loading.py       # helpers to load/prepare thermal images
│   ├── fusion_model.py          # simple late-fusion baseline
│   └── run_pipeline.py          # entry point to reproduce the experiment
├── figures/
│   ├── table_2.png
│   └── confusion_matrix_test_percent.png
├── data/
│   └── (place your CSVs / images here)
├── requirements.txt
└── LICENSE
```

## How to run (local)

1. Create and activate a virtual environment (optional but recommended). Example:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your **tabular CSV** and your **thermal images** under `data/`. Adjust the paths in `src/data_loading.py` and `src/imaging_loading.py` if your structure is different.

4. Run the full pipeline (tabular -> imaging -> fusion):

   ```bash
   python -m src.run_pipeline --config configs/default.yaml
   ```

   or, more simply:

   ```bash
   python -m src.run_pipeline
   ```

## Notes on leakage and splits

The notebook already enforces a **strict past-only + gap** split and a **per-cow / per-visit** logic to avoid overly optimistic results. Please keep this behaviour if you modify or extend the work, especially before publishing the results.

## Colab

If you wish to run it again on Colab:

- upload the content of `notebooks/`;
- mount your Drive;
- adjust the paths in the first cells so that they point to `/content/...` as in the original notebook;
- re-run the pipeline end-to-end.


---
If you have received this repository as part of a paper submission, you should be able to reproduce the figures in `figures/` directly from the notebook.
