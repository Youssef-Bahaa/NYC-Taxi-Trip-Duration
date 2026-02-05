
# NYC Taxi Trip Duration — Tutorial

A step-by-step guide to reproduce experiments and get a submission.

## 1. Setup
- Install Python 3.8+ and create a virtualenv:
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

## 2. Data
- Download Kaggle dataset and place files:
  data/raw/train.csv
  data/raw/test.csv

## 3. Preprocessing
- Run:
  python src/preprocess.py \
    --train data/raw/train.csv \
    --test data/raw/test.csv \
    --out data/processed/

- What it does:
  - Parses datetimes, fills missing values
  - Adds features: hour, dow, haversine_distance, bearing
  - Saves parquet files in `data/processed/`


## Files to inspect
- `notebooks/` — walkthroughs
- `src/preprocess.py` — transformation pipeline
- `src/train.py` — training configs and CV logic
