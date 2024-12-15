<<<<<<< HEAD

# Brist1D Project

This repository contains scripts and models for the Brist1D project. It includes preprocessing steps, 
model training scripts, and output predictions for various machine learning approaches.

## Project Structure

- `preprocessing-steps.py`: Scripts for data preprocessing.
- `models/`: Contains various model scripts, including LightGBM, LSTM, and ensembles.
- `outputs/`: Includes CSV files with predictions and submission files.
- `train.csv`, `test.csv`: Datasets used for training and testing.
- `catboost_info/`: Logs and metadata for CatBoost training.

## Getting Started

### Prerequisites

Make sure you have Python 3.8 or higher installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Models

1. Preprocess the data using `preprocessing-steps.py`.
2. Train models using scripts in the `models/` directory.
3. Generate predictions and evaluate performance.

### Example

```bash
python models/brist1d-lgbm.py
```

## Outputs

The `outputs/` directory contains sample predictions and submissions for benchmarking.
