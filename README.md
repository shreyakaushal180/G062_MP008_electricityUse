# G062 | MP008: Household Electricity Use Prediction

**Course:** SCSE3040 MLOps Mini Project  
**Group:** G062  
**Problem:** MP008 - Household Electricity Use (Lucknow)  
**Faculty:** Dr. Avantika Saklani  
**Batch:** 90

## Team

| Enrollment number | Member |
|---|---|
| S24CSEU0417 | Shreya Kaushal |
| S24CSEU0185 | Vishakha |
| S24CSEU0601 | Naivedya Shilpi |
| S24CSEU1288 | Abhirup Das |

## Project objective

Build an end-to-end MLOps pipeline to predict household electricity consumption (`units_kwh`) for households in Lucknow. The assignment covers data quality investigation and cleaning, DVC data versioning, model training and MLflow tracking, model registration and promotion, data-drift monitoring, retraining, automated tests, and holdout predictions.

## Repository structure

```text
.
├── .dvc/
├── .dvcignore
├── .gitignore
├── README.md
├── mlflow.db
├── mlruns/
├── params.yaml
├── requirements.txt
├── data/
│   └── raw/
│       ├── holdout_features.csv
│       ├── train.csv.dvc
│       └── later.csv.dvc
├── models/
│   └── model.joblib
├── predictions/
│   └── holdout_predictions.csv
├── reports/
│   ├── findings.yaml
│   ├── model_comparison.json
│   ├── v1_metrics.json
│   ├── v2_metrics.json
│   ├── feeder_metrics.json
│   └── promotion_decision.json
├── scripts/
│   ├── setup_dvc.ps1
│   ├── train.py
│   ├── evaluate.py
│   └── generate_predictions.py
├── src/
│   └── mp008/
│       ├── __init__.py
│       ├── cleaning.py
│       ├── data_checks.py
│       ├── evaluation.py
│       ├── mlflow_utils.py
│       ├── monitoring.py
│       ├── preprocessing.py
│       └── training.py
└── tests/
    ├── test_mlflow.py
    ├── test_predictions.py
    └── test_training.py
```

## Prerequisites

- Windows PowerShell
- Python 3.12+ (or the course virtual environment)
- Git
- DVC

## 1. Installing requirements

From the repository root in Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
```

## 2. Obtaining and verifying DVC data

Place the original faculty-provided CSV files at:

```text
data/raw/train.csv
data/raw/later.csv
data/raw/holdout_features.csv
```

Do not open and save `train.csv` in Excel. Verify the SHA-256 fingerprints and initialize DVC pointers from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_dvc.ps1
```

If a shared DVC remote is configured, retrieve the tracked datasets with:

```powershell
dvc pull
```

Review the DVC pointer files:

```powershell
Get-Content .\data\raw\train.csv.dvc
Get-Content .\data\raw\later.csv.dvc
```

## 3. Running data checks and data cleaning

Run the Phase 1 data quality investigation script:

```powershell
python -m src.mp008.data_checks --train data/raw/train.csv --later data/raw/later.csv --holdout data/raw/holdout_features.csv --findings reports/findings.yaml
```

The cleaning module (`src/mp008/cleaning.py`) provides:
- `clean_training_data(raw_df)` -> returns `(X, y)` after removing exact duplicates, converting `Feeder-1` `ac_hours_per_day` from minutes to hours (`/ 60`), correcting the 12 `10x` target typos in `units_kwh`, and excluding the post-event leakage column `billed_units_after_reading`.
- `clean_inference_data(raw_df)` -> selects and cleans the 8 approved prediction features (`rooms`, `members`, `ac_hours_per_day`, `avg_temp_c`, `house_area_sqft`, `solar_kw`, `connection`, `feeder`).

## 4. Training V1 and V2 models (with MLflow tracking & registration)

All configuration settings (paths, expected SHA-256 hashes, candidate hyperparameters, random seed, MLflow settings, and output locations) are stored in `params.yaml`.

Run the complete training, evaluation, registration, and promotion pipeline:

```powershell
python scripts/train.py --params params.yaml
```

What `scripts/train.py` executes:
1. **V1 Training & Candidate Sweep:** Cleans `data/raw/train.csv` (1000 rows) and trains three candidate scikit-learn pipelines (`RandomForestRegressor`, `ExtraTreesRegressor`, `HistGradientBoostingRegressor`) with `ColumnTransformer` preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`).
2. **V1 Evaluation:** Evaluates every V1 candidate on the complete `data/raw/later.csv` dataset (300 rows) using MAE, RMSE, and R², and computes feeder-wise MAE (`Feeder-1`, `Feeder-2`, `Feeder-3`).
3. **V1 Registration:** Logs all runs to `sqlite:///mlflow.db` under experiment `MP008_Electricity_Consumption` with `data_sha256` (`9aea26c5aa76831f4a4e58a3dfafd06d47144136c13ace14834010bfcf472781`) and registers the best V1 model (`HistGradientBoostingRegressor`) as **Version 1** of `MP008_Electricity_Model`.
4. **V2 Chronological Retraining:** Sorts `data/raw/later.csv` chronologically by `date` ascending using stable mergesort (preserving record arrival order `record_id` within each calendar date), and applies the faculty-required 70/30 row-based split (`PROBLEM.md` Section 7): the first 70% (210 rows, `2026-07-01` to `2026-08-14`, records `R20001`–`R20210`) is used for additional training and the final 30% (90 rows, `2026-08-14` to `2026-08-31`, records `R20211`–`R20300`) is held out for V2 evaluation. Note that the boundary date `2026-08-14` has its first 3 chronological records in the 70% training split and its remaining 6 chronological records in the 30% evaluation split (`split_later_chronologically` also supports `date_grouped=True` if an unsplit calendar-date boundary of 207/93 rows at `2026-08-13` / `2026-08-14` is desired). Retrains V2 on cleaned `train.csv` + first 70% of `later.csv` (1210 rows total) and registers it as **Version 2** of `MP008_Electricity_Model`.
5. **Promotion Rule & Evaluation Protocol Note:** Per the project protocol, V1 is evaluated on the complete `later.csv` dataset (`MAE = 98.0774`) as well as on the chronological final 30% check segment (`MAE = 93.3714`), while V2 is evaluated on the chronological final 30% check segment (`MAE = 42.2941`). Because the complete `later.csv` and its final 30% segment are different evaluation splits, both comparisons are recorded in `reports/promotion_decision.json`. Under both comparisons, V2 reduces MAE by more than 54% (well above the required 10% threshold `V2_MAE <= V1_MAE * 0.90`), so Version 2 is promoted to `@champion` and saved to `models/model.joblib`.

## 5. Evaluating the saved champion model and drift monitoring

To run standalone evaluation and Population Stability Index (PSI) drift checks on the saved champion model (`models/model.joblib`):

```powershell
python scripts/evaluate.py --params params.yaml
```

The PSI calculation (`src/mp008/monitoring.py`) derives 10-bin reference quantile boundaries from cleaned `train.csv` while deduplicating repeated quantile edges for discrete features (`rooms` -> 6 effective bins, `members` -> 8 effective bins) and safely handling constant features so zero-width bins never occur.

## 6. Running the MLflow UI and loading registered model artifacts portably

Both `mlflow.db` and `mlruns/` are tracked in the repository, and all artifact URIs stored in `mlflow.db` (`experiments`, `runs`, `logged_models`, and `model_versions`) are normalized to relative `mlruns/1/...` paths by `make_artifact_paths_portable()`. This ensures that any fresh clone on any machine can load both registered model versions (`models:/MP008_Electricity_Model/1` and `models:/MP008_Electricity_Model/2`) and the `@champion` alias (`models:/MP008_Electricity_Model@champion`) directly from the repository root without machine-specific paths.

To inspect all logged runs, parameters (`data_sha256`), metrics, model artifacts, and registered model versions (`MP008_Electricity_Model` Versions 1 and 2, with `@champion` alias) from the repository root:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open `http://127.0.0.1:5000` in your browser.

To verify loading the registered models and `@champion` alias in Python from the repository root:

```powershell
python -c "import mlflow, mlflow.sklearn; mlflow.set_tracking_uri('sqlite:///mlflow.db'); m = mlflow.sklearn.load_model('models:/MP008_Electricity_Model@champion'); print(type(m))"
```

## 7. Generating holdout predictions

Generate predictions for `data/raw/holdout_features.csv` (250 rows) using the champion model (`models/model.joblib`):

```powershell
python scripts/generate_predictions.py --params params.yaml
```

This writes `predictions/holdout_predictions.csv` containing exact columns `record_id,prediction` and 250 non-negative numeric predictions (clipped at a minimum of `0.0` kWh).

## 8. Running automated tests

Run the automated test suite with `pytest`:

```powershell
python -m pytest -v
```


## 9. Prediction API, logging, CI and Kubernetes

### Prediction API (`app/main.py`)

- `GET /health` returns `{"status": "ok", "model_loaded": true}`.
- `POST /predict` takes exactly these fields: `rooms`, `members`, `ac_hours_per_day`, `avg_temp_c`, `house_area_sqft`, `solar_kw`, `connection`, `feeder`, and returns `predicted_units_kwh`.
- The client's two rules are enforced in the pydantic schema, so a bad request gets status 422 and never reaches the model:
  - `solar_kw` must be at most 9.
  - `house_area_sqft` must be at least 150.

Run locally:

```powershell
python -m uvicorn app.main:app --port 8000
```

### JSON logging (`app/logging_config.py`)

Every prediction writes one JSON line to stdout (never to a file), with the timestamp, level, feeder, predicted value and latency. No personal details are logged.

### Docker

```powershell
docker build -t mp008-api:1.0 .
docker run -p 8000:8000 mp008-api:1.0
```

### Continuous integration (`.github/workflows/ci.yml`)

On every push, GitHub Actions installs `requirements.txt` and runs `pytest`. The API tests (`tests/test_api.py`) use FastAPI's `TestClient` with a stub model, so they need no data files.

### Kubernetes (`k8s/`)

- `deployment.yaml`: Deployment `mp008-api`, 2 replicas, image `mp008-api:1.0`, container port 8000. Readiness and liveness probes both call `/health`, and liveness waits longer (20 s against 5 s).
- `service.yaml`: Service `mp008-svc`, port 80 forwarding to container port 8000. Its selector matches the Pod label `app: mp008-api`.

```powershell
kubectl apply -f k8s/
```
