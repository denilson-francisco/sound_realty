"""
Evaluate the currently saved model against labeled data and log the results.

This script is NOT for retraining. Its purpose is drift monitoring: load the
model that is currently in production, measure its performance on a labeled
dataset, and record the results so they can be compared against past training
runs over time.

Run periodically with new labeled data to detect when the model starts
degrading and a retrain is needed:

    python evaluate_model.py

Each run logs to MLflow under the same experiment as create_model.py, tagged
as run_type=evaluation. A JSON snapshot is also saved to data/training_results/
with a training_run_id field that links back to the create_model.py run that
produced the model being evaluated.
"""

import datetime
import json
import pathlib
import pickle

import mlflow
import numpy
import pandas
from sklearn import metrics
from sklearn import model_selection

SALES_PATH = "data/kc_house_data_new_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMN_SELECTION = [
    "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "zipcode", "lat", "long",
    "sqft_living15", "sqft_lot15",
]
MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/model_features.json"
METADATA_PATH = "model/model_metadata.json"
RESULTS_DIR = "data/training_results"


def load_data(sales_path: str, demographics_path: str) -> tuple:
    """Load and merge sales and demographics data for evaluation.

    Args:
        sales_path: path to CSV file with labeled home sale data
        demographics_path: path to CSV file with demographic data

    Returns:
        Tuple with two elements: a DataFrame of features and a Series of
        target prices, both aligned by index.
    """
    data = pandas.read_csv(sales_path, usecols=SALES_COLUMN_SELECTION,
                           dtype={"zipcode": str})
    demographics = pandas.read_csv(demographics_path, dtype={"zipcode": str})
    merged = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged.pop("price")
    return merged, y


def rmse(y_true, y_pred) -> float:
    """Compute root mean squared error between true and predicted values.

    Args:
        y_true: array of true target values
        y_pred: array of predicted target values

    Returns:
        Root mean squared error as a float
    """
    return numpy.sqrt(metrics.mean_squared_error(y_true, y_pred))


def main():
    """Load the saved model, evaluate it on labeled data, and log the results."""
    print("Loading data...")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.30, random_state=42)

    print(f"Dataset: {len(x):,} rows  |  Train: {len(x_train):,}  |  Test: {len(x_test):,}\n")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH) as f:
        features = json.load(f)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    x_test_eval = x_test[features]

    y_pred = model.predict(x_test_eval)
    eval_r2 = round(metrics.r2_score(y_test, y_pred), 4)
    eval_mae = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    eval_rmse = round(rmse(y_test, y_pred), 2)

    print("Held-out test set performance")
    print(f"  R²  : {eval_r2:.4f}")
    print(f"  MAE : ${eval_mae:>12,.0f}")
    print(f"  RMSE: ${eval_rmse:>12,.0f}")

    run_timestamp = datetime.datetime.now(datetime.timezone.utc)
    run_name = f"evaluation_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_experiment("sound-realty-price-prediction")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", "evaluation")
        mlflow.set_tag("training_run_id", metadata["mlflow_run_id"])
        mlflow.log_param("model_type", metadata["model_type"])
        mlflow.log_param("evaluated_on", SALES_PATH)
        mlflow.log_param("n_samples", len(x_test_eval))
        mlflow.log_metric("r2", eval_r2)
        mlflow.log_metric("mae", eval_mae)
        mlflow.log_metric("rmse", eval_rmse)
        run_id = run.info.run_id

    results_dir = pathlib.Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "mlflow_run_id": run_id,
        "mlflow_run_name": run_name,
        "run_type": "evaluation",
        "timestamp": run_timestamp.isoformat(),
        "training_run_id": metadata["mlflow_run_id"],
        "model_type": metadata["model_type"],
        "evaluated_on": SALES_PATH,
        "n_samples": len(x_test_eval),
        "metrics": {
            "r2": eval_r2,
            "mae": eval_mae,
            "rmse": eval_rmse,
        },
    }
    json.dump(snapshot, open(results_dir / f"{run_name}.json", "w"), indent=2)

    print(f"\nRun snapshot saved to {results_dir}/{run_name}.json")
    print(f"MLflow run ID: {run_id}  (run 'mlflow ui' to explore)")


if __name__ == "__main__":
    main()
