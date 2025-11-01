"""Entry point to reproduce the multimodal mastitis pipeline.
This is intentionally lightweight: the full logic is in the notebook.
"""

import argparse
import yaml
from pathlib import Path

from .data_loading import load_tabular_data
from .fusion_model import LateFusionModel


def main():
    parser = argparse.ArgumentParser(description="Run multimodal mastitis detection pipeline.")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg["tabular"]["csv_file"]
    target_column = cfg["tabular"]["target_column"]
    X_train, X_val, X_test, y_train, y_val, y_test = load_tabular_data(csv_path, target_column)

    # Here you would:
    # 1. train a tabular model (e.g. XGBoost / LightGBM)
    # 2. extract imaging embeddings from your CNN
    # 3. fuse using LateFusionModel
    # For the GitHub version we just show that the pipeline runs.

    print("Loaded tabular data:")
    print("  train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

    fusion = LateFusionModel(imaging_weight=cfg["fusion"]["imaging_weight"])
    print("Fusion model created with imaging weight =", fusion.imaging_weight)

    print("Pipeline finished (stub). Please refer to the notebook for the full experiment.")


if __name__ == "__main__":
    main()
