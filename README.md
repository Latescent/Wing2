# Bee Wing Morphology Analysis Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

This repository contains the complete pipeline for analyzing bee wing vein structures, including:
- Image preprocessing and skeletonization
- Vein intersection detection
- Anomaly detection (Isolation Forest + Autoencoder)
- Morphological clustering

## Reproduction Guide
1. Install dependencies: `pip install -r requirements.txt`
2. Download datasets: `scripts/download_data.sh`
3. Run full pipeline: `scripts/run_pipeline.sh`

## Key Features
✔️ Region-specific parameter optimization (`configs/tweaks.csv`)  
✔️ Parallel processing for large datasets  
✔️ Interactive Jupyter Notebook tutorials  
✔️ Convolutional Autoencoder for unsupervised anomaly detection

## Dataset Information
- **26,000 raw wing images** (Zenodo: 10.5281/zenodo.XXXXX)
- **1,000 annotated samples** (`data/annotated/`)
- **Parameter configurations** (`configs/tweaks.csv`)

[Full Documentation](docs/reproduction-guide.md) | [Methodology](docs/methodology.md)