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
├── .dockerignore
├── .dvc/
├── .dvcignore
├── .gitignore
├── Dockerfile
├── README.md
├── mlflow.db
├── mlruns/
├── params.yaml
├── requirements-api.txt
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
│       ├── api.py
│       ├── cleaning.py
│       ├── data_checks.py
│       ├── evaluation.py
│       ├── mlflow_utils.py
│       ├── monitoring.py
│       ├── preprocessing.py
│       └── training.py
└── tests/
    ├── test_api.py
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

Run the full automated test suite (Phase 2 + Phase 3 API tests) with `pytest`:

```powershell
python -m pytest -v
```

Run only the Phase 3 FastAPI `TestClient` suite:

```powershell
python -m pytest tests/test_api.py -v
```

## 9. Phase 3: FastAPI Prediction Service & Docker Deployment

### API Architecture

- **Module:** `src/mp008/api.py` (`uvicorn src.mp008.api:app`)
- **Model Artifact:** Loads `models/model.joblib` relative to the project root (independent of current working directory).
- **Preprocessing Consistency:** Applies `clean_inference_data()` from `src/mp008/cleaning.py` before passing the 8 approved features in exact training order to the scikit-learn pipeline.
- **Non-Negative Guard:** Clips predictions at `0.0` kWh (`max(0.0, raw_prediction)`).
- **Structured JSON Logging:** Emits single-line JSON log events to `stdout` (`timestamp`, `level`, `event`, `request_id`, `duration_ms`) for startup, shutdown, successful predictions, validation failures, and errors without logging full payloads.

### Endpoints

#### `GET /health`
- **Status:** `200 OK`
- **Response:**
```json
{"status": "ok"}
```

#### `POST /predict`
Accepts the 8 approved prediction features (`record_id`, `date`, `billed_units_after_reading`, and `units_kwh` are not accepted):

| Field | Type | Validation Rules |
|---|---|---|
| `rooms` | `int` | Required, strict integer, `> 0` |
| `members` | `int` | Required, strict integer, `> 0` |
| `ac_hours_per_day` | `float` | Required, finite number, `0 <= ac_hours_per_day <= 24` |
| `avg_temp_c` | `float` | Required, finite number |
| `house_area_sqft` | `float` | Required, finite number, `house_area_sqft >= 150` (**values `< 150` return HTTP 422**) |
| `solar_kw` | `float` | Required, finite number, `0 <= solar_kw <= 9` (**values `> 9` return HTTP 422**) |
| `connection` | `str` | Required, non-empty string |
| `feeder` | `str` | Required, non-empty string |

Unexpected extra fields and non-numeric types (such as strings or booleans for numerical fields) are rejected with **HTTP 422**.

#### Example Valid Request

```json
{
  "rooms": 4,
  "members": 5,
  "ac_hours_per_day": 6.5,
  "avg_temp_c": 31.2,
  "house_area_sqft": 1800.0,
  "solar_kw": 3.5,
  "connection": "domestic_plus",
  "feeder": "Feeder-2"
}
```

#### Example Prediction Response (`200 OK`)

```json
{
  "prediction": 442.19,
  "predicted_units_kwh": 442.19,
  "unit": "kWh"
}
```

### Running the API Locally

From the repository root in Windows PowerShell:

```powershell
python -m uvicorn src.mp008.api:app --host 0.0.0.0 --port 8000
```

Verify endpoints in another PowerShell window:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

$body = '{"rooms":4,"members":5,"ac_hours_per_day":6.5,"avg_temp_c":31.2,"house_area_sqft":1800.0,"solar_kw":3.5,"connection":"domestic_plus","feeder":"Feeder-2"}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

### Docker Build & Run

The `Dockerfile` uses `python:3.12-slim`, installs the lean serving dependencies from `requirements-api.txt` (omitting training/tracking-only packages such as MLflow, PyArrow, and Matplotlib while keeping `PyYAML`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib`, `fastapi`, `uvicorn`, and `pydantic` at exact pinned versions matching `requirements.txt`) prior to copying `src/`, `models/model.joblib`, and `params.yaml`, exposes port `8000`, and starts `uvicorn src.mp008.api:app --host 0.0.0.0 --port 8000`.

Build the container image:

```powershell
docker build -t mp008-api:1.0 .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 mp008-api:1.0
```
## 10. Continuous Integration and Kubernetes Deployment

### GitHub Actions CI

The GitHub Actions workflow is located at `.github/workflows/ci.yml`.

It runs automatically on pushes and pull requests. The workflow:
1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs dependencies from `requirements.txt`.
4. Runs the complete automated test suite using `pytest`.

### Kubernetes Deployment

The Kubernetes configuration is located in the `k8s/` directory.

- `k8s/deployment.yaml`: Creates the `mp008-api` Deployment with 2 replicas, using image `mp008-api:1.0` and container port 8000.
- `k8s/service.yaml`: Creates the `mp008-svc` ClusterIP Service, exposing port 80 and forwarding traffic to port 8000.
- Readiness and liveness probes use the `/health` endpoint. The liveness probe has a longer initial delay.

Apply the Kubernetes configuration:

```powershell
kubectl apply -f k8s/
```

Check deployment status:

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
```

The health endpoint returns exactly:

```json
{"status": "ok"}
```
