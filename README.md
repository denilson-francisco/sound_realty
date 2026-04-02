# Sound Realty — Home Price Prediction API

This service exposes a REST API for predicting home sale prices in the Seattle area.
It wraps a scikit-learn model trained on King County sales data, enriches requests
with census demographics on the backend, and is deployed via Docker Swarm for
horizontal scaling and zero-downtime model updates.

---

## Project structure

```
api/
  main.py                  FastAPI application
  Dockerfile               builds the API image
  requirements.txt         API dependencies
data/
  kc_house_data.csv        training data
  zipcode_demographics.csv census demographics joined at prediction time
  future_unseen_examples.csv  sample inputs for testing
  training_results/        JSON snapshot per training run
model/
  model.pkl                trained model artifact
  model_features.json      feature list in training order
  model_percentiles.json   p5/p95 bounds per feature (for warnings)
  model_defaults.json      training-set medians (for /predict/basic imputation)
create_model.py            train and log a model
evaluate_model.py          evaluate the deployed model on labeled data (drift monitoring)
test_api.py                demo client
docker-compose.yml         Swarm deployment config
conda_environment.yml      Python environment
```

---

## Setup

```bash
conda env create -f conda_environment.yml
conda activate housing
```

---

## Training

**Baseline** — reproduces the original 8-feature KNN as the first tracked run:
```bash
python create_model.py --baseline
```

**Improved** — runs all candidates in `CANDIDATE_MODELS` (currently KNN and Random Forest),
selects the winner by 5-fold cross-validation on the train set, evaluates once on the
held-out test set:
```bash
python create_model.py
```

To add a new model candidate, append a tuple to `CANDIDATE_MODELS` in `create_model.py`.
The comparison, selection, logging, and JSON snapshot are all automatic.

Training artifacts are saved to `model/`. Every run is logged to MLflow:
```bash
mlflow ui   # open http://localhost:5000
```

---

## Drift monitoring

To evaluate the currently deployed model on new labeled data without retraining:
```bash
python evaluate_model.py
```

Each run is logged to the same MLflow experiment so R² trend is visible over time.
The JSON snapshot in `data/training_results/` includes a `training_run_id` field
that links back to the training run that produced the model being evaluated.

---

## API

### Versioning

Images follow `MAJOR.MINOR.PATCH`:

| Change | Version bump |
|---|---|
| Input/output schema change, model family change | MAJOR |
| Retrain with more data, better features, better hyperparameters | MINOR |
| Bug fix in preprocessing, metadata correction | PATCH |

### Endpoints

**`GET /health`**
Returns `{"status": "ok"}`. Used by Docker Swarm for health checks.

**`POST /predict`**
Accepts the 18 columns from `future_unseen_examples.csv`. Demographics are joined
server-side using `zipcode` — they must not be included in the request.

Returns:
```json
{
  "prediction_id": "uuid",
  "predicted_price": 485000.0,
  "warnings": [],
  "data_quality_score": 1.0,
  "model_name": "sound-realty-price-predictor",
  "model_version": "1.1.0",
  "model_stage": "development",
  "schema_version": "1.0",
  "prediction_latency_ms": 12.4,
  "timestamp": "2026-04-02T..."
}
```

`warnings` lists any out-of-range feature values (outside p5/p95 training bounds)
or cross-field inconsistencies (e.g. `sqft_above + sqft_basement != sqft_living`).
`data_quality_score` starts at 1.0 and decreases by 0.1 per warning.

**`POST /predict/basic`**
Requires only the 8 core house features plus `zipcode`. The remaining 10 features
are optional — if not provided, training-set medians are used. Each imputed field
is listed in the `warnings` field of the response.

Required fields: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`,
`sqft_above`, `sqft_basement`, `zipcode`.

---

## Deployment

### Build

```bash
# Initial build (version defaults to 1.0.0 from Dockerfile ARG)
docker build -f api/Dockerfile -t sound-realty-api:1.0.0 .

# Build a new version
docker build -f api/Dockerfile --build-arg MODEL_VERSION=1.1.0 -t sound-realty-api:1.1.0 .
```

Model artifacts are baked into the image at build time. The image tag is the
model version — no shared volumes, no runtime artifact loading from outside the container.

### Deploy with Docker Swarm

```bash
docker swarm init
docker stack deploy -c docker-compose.yml realty
```

The stack runs 2 replicas by default. Swarm polls `/health` every 30 seconds and
replaces any replica that fails 3 consecutive checks.

### Zero-downtime model update

```bash
docker build -f api/Dockerfile --build-arg MODEL_VERSION=1.1.0 -t sound-realty-api:1.1.0 .
docker service update --image sound-realty-api:1.1.0 realty_api
```

`docker-compose.yml` is configured with `order: start-first` — the new replica
is started and health-checked before the old one is stopped, so there is no
gap in service during the rollout.

### Scaling

```bash
docker service scale realty_api=4   # scale up
docker service scale realty_api=2   # scale down
```

Scaling does not restart running replicas.

---

## Testing

### Prerequisites

The test commands below use an explicit `--url` flag. On a standard Linux or macOS setup
`http://localhost:8000` works directly. On Windows with Docker running inside WSL, the
API is bound to the WSL network interface rather than the Windows loopback, so the WSL
IP must be used instead. Find it with:

```bash
hostname -I   # run inside WSL
```

Replace `<WSL_IP>` in the commands below with the first address returned.

### Test commands

```bash
mkdir logs

# Phase 1 — train baseline model
python create_model.py --baseline > logs/train_baseline.txt

# Phase 2 — build and deploy version 1.0.0
docker build -f api/Dockerfile -t sound-realty-api:1.0.0 .
docker swarm init
docker stack deploy -c docker-compose.yml realty
docker service ls > logs/swarm_deploy.txt

# Phase 3 — verify the API is alive
curl -v http://<WSL_IP>:8000/health

# Phase 4 — test baseline predictions
python test_api.py --url http://<WSL_IP>:8000 > logs/test_api_baseline.txt

# Phase 5 — evaluate baseline model (drift monitoring baseline)
python evaluate_model.py > logs/evaluate_baseline.txt

# Phase 6 — train improved model (KNN vs Random Forest)
python create_model.py > logs/train_improved.txt

# Phase 7 — rolling update to version 1.1.0
docker build -f api/Dockerfile --build-arg MODEL_VERSION=1.1.0 -t sound-realty-api:1.1.0 .
docker service update --image sound-realty-api:1.1.0 realty_api
docker service ps realty_api > logs/rolling_update.txt

# Phase 8 — verify update (response should show model_version 1.1.0)
python test_api.py --url http://<WSL_IP>:8000 > logs/test_api_improved.txt

# Phase 9 — evaluate improved model
python evaluate_model.py > logs/evaluate_improved.txt

# Phase 10 — scaling demo
docker service scale realty_api=4
docker service ls > logs/scale_up.txt
docker service scale realty_api=2
docker service ls > logs/scale_down.txt

# Bonus — basic endpoint with 8 fields only
python test_api.py --basic --url http://<WSL_IP>:8000 > logs/test_api_basic.txt
```
