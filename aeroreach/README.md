# AeroReach Customer Insight Project

## Structure
- `data/`: Data loading utilities
- `preprocessing/`: Data cleaning, outlier handling, scaling
- `clustering/`: K-Prototypes and Hierarchical Gower clustering
- `classification/`: Random Forest segment classifier
- `evaluation/`: Metrics for clustering and classification
- `utils/`: Feature config and encoding helpers
- `main_pipeline.py`: End-to-end pipeline script

## How to Run
1. Install dependencies:
   ```
pip install -r requirements.txt
   ```
2. Run the main pipeline:
   ```
python aeroreach/main_pipeline.py
   ```

## Description
This project implements clustering and classification for targeted tourism marketing using the AeroReach Insights dataset. It supports mixed-type clustering (K-Prototypes, Hierarchical Gower) and segment classification (Random Forest).

See `SYNOPSIS.txt` for methodology details.
