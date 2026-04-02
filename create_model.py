import argparse
import datetime
import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import mlflow
import numpy
import pandas
from sklearn import base
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
SALES_COLUMN_SELECTION_BASELINE = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode',
]
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
    'sqft_living15', 'sqft_lot15',
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved
RESULTS_DIR = "data/training_results"  # Directory where JSON run snapshots are saved

# Registry of candidate models for improved-mode comparison.
# To add a new model: append a (name, estimator) tuple here.
# Everything else — CV, selection, logging, JSON snapshot — is automatic.
CANDIDATE_MODELS = [
    (
        "KNN",
        pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            neighbors.KNeighborsRegressor(),
        ),
    ),
    (
        "RandomForest",
        ensemble.RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1,
        ),
    ),
]


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with demographics data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple with two elements: a DataFrame and a Series of the same length.
        The DataFrame contains features for machine learning, the series
        contains the target variable (home sale price).
    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path, dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def cross_validate_estimator(name: str, estimator, x: pandas.DataFrame,
                              y: pandas.Series, cv: int = 5) -> dict:
    """Run k-fold cross-validation on an unfitted estimator and return scores.

    Model selection is based on CV mean R² rather than test set performance,
    which ensures the held-out test set remains uncontaminated until final
    evaluation.

    Args:
        name: label used to identify the estimator in printed output
        estimator: an unfitted scikit-learn estimator or pipeline
        x: feature DataFrame to cross-validate on
        y: target Series aligned with x
        cv: number of folds for cross-validation

    Returns:
        Dict with keys cv_fold_scores (list), cv_mean_r2, and cv_std_r2.
    """
    fold_scores = model_selection.cross_val_score(
        estimator, x, y, cv=cv, scoring="r2", n_jobs=-1)
    cv_mean = round(float(fold_scores.mean()), 4)
    cv_std = round(float(fold_scores.std()), 4)
    print(f"  {name:<30} CV mean R²={cv_mean:.4f}  std={cv_std:.4f}  "
          f"folds={[round(s, 4) for s in fold_scores]}")
    return {
        "cv_fold_scores": [round(float(s), 4) for s in fold_scores],
        "cv_mean_r2": cv_mean,
        "cv_std_r2": cv_std,
    }


def compute_test_metrics(model, x_test: pandas.DataFrame,
                         y_test: pandas.Series) -> dict:
    """Compute final held-out test metrics for the selected model.

    This should be called only once, after model selection is complete,
    to avoid any test set contamination.

    Args:
        model: a fitted scikit-learn estimator with a predict method
        x_test: held-out feature DataFrame never used during training or
            model selection
        y_test: target Series corresponding to x_test

    Returns:
        Dict with keys test_r2, test_mae, and test_rmse.
    """
    y_pred = model.predict(x_test)
    return {
        "test_r2": round(metrics.r2_score(y_test, y_pred), 4),
        "test_mae": round(metrics.mean_absolute_error(y_test, y_pred), 2),
        "test_rmse": round(float(numpy.sqrt(
            metrics.mean_squared_error(y_test, y_pred))), 2),
    }


def compute_percentiles(x_train: pandas.DataFrame) -> dict:
    """Compute p5 and p95 percentiles for each feature in the training set.

    Args:
        x_train: feature DataFrame used to train the model

    Returns:
        Dict mapping each feature name to a dict with keys p5 and p95,
        used by the API to generate out-of-range warnings at prediction time.
    """
    percentiles = {}
    for col in x_train.columns:
        percentiles[col] = {
            "p5": round(float(x_train[col].quantile(0.05)), 4),
            "p95": round(float(x_train[col].quantile(0.95)), 4),
        }
    return percentiles


def sanitize_params(params: dict) -> dict:
    """Convert estimator.get_params() output to JSON-serializable primitives.

    scikit-learn pipelines include nested estimator objects in get_params(),
    which are not JSON-serializable. This reduces them to their repr string so
    the full parameter snapshot can be written to disk.

    Args:
        params: dict returned by estimator.get_params()

    Returns:
        Dict with the same keys where any non-primitive value is replaced by
        its str() representation.
    """
    result = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def get_feature_importances(model, feature_names: list, top_n: int = 10) -> dict:
    """Extract the top feature importances from a Random Forest model.

    Args:
        model: a fitted RandomForestRegressor
        feature_names: list of feature names in the order passed during training
        top_n: number of top features to return

    Returns:
        Dict mapping feature name to importance score, sorted descending,
        limited to top_n entries. Returns an empty dict for models that do
        not expose feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        return {}
    importances = dict(zip(feature_names, model.feature_importances_))
    sorted_importances = dict(
        sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n]
    )
    return {k: round(v, 4) for k, v in sorted_importances.items()}


def main():
    """Load data, compare candidate models, export the best one, and log the run.

    Accepts an optional --baseline flag. Without it, runs the improved pipeline
    that compares KNN against Random Forest using all 18 sales features. With
    --baseline, reproduces the original model (8 features, KNN only) while
    still logging the full tracking metadata.

    Model selection is done via cross-validation on the train_val set. The
    held-out test set is only evaluated once, after the winner is chosen.
    """
    parser = argparse.ArgumentParser(
        description="Train and track a Sound Realty home price prediction model.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train the original 8-feature KNN model without model comparison",
    )
    args = parser.parse_args()

    column_selection = (SALES_COLUMN_SELECTION_BASELINE
                        if args.baseline else SALES_COLUMN_SELECTION)
    mode = "baseline" if args.baseline else "improved"
    print(f"Mode: {mode}\n")

    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, column_selection)
    x_train_val, x_test, y_train_val, y_test = model_selection.train_test_split(
        x, y, test_size=0.30, random_state=42)

    print(f"train_val: {len(x_train_val):,} rows  |  test (held out): {len(x_test):,} rows\n")

    print("Cross-validating on train_val (5 folds):")

    if args.baseline:
        knn_estimator = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            neighbors.KNeighborsRegressor(),
        )
        knn_cv = cross_validate_estimator("KNN", knn_estimator, x_train_val, y_train_val)
        best_estimator = knn_estimator
        selected_name = "KNN"
        models_list = [
            {"model_type": "KNN", "hyperparameters": sanitize_params(knn_estimator.get_params()),
             **knn_cv, "selected": True},
        ]
        cv_results = [{"name": "KNN", **knn_cv}]
        print("\nBaseline KNN selected.")
    else:
        cv_results = []
        for name, estimator in CANDIDATE_MODELS:
            cv = cross_validate_estimator(name, estimator, x_train_val, y_train_val)
            cv_results.append({"name": name, "estimator": estimator, **cv})

        best = max(cv_results, key=lambda r: r["cv_mean_r2"])
        best_estimator = best["estimator"]
        selected_name = best["name"]
        print(f"\n{selected_name} selected (CV mean R² {best['cv_mean_r2']:.4f}).")

        models_list = [
            {
                "model_type": r["name"],
                "hyperparameters": sanitize_params(r["estimator"].get_params()),
                "cv_fold_scores": r["cv_fold_scores"],
                "cv_mean_r2": r["cv_mean_r2"],
                "cv_std_r2": r["cv_std_r2"],
                "selected": r["name"] == selected_name,
            }
            for r in cv_results
        ]

    print(f"\nFitting {selected_name} on full train_val set...")
    best_model = base.clone(best_estimator).fit(x_train_val, y_train_val)

    print("Evaluating on held-out test set (run once):")
    test_metrics = compute_test_metrics(best_model, x_test, y_test)
    cv_winner = next(m for m in models_list if m["selected"])
    cv_test_gap = round(cv_winner["cv_mean_r2"] - test_metrics["test_r2"], 4)
    print(f"  test R²={test_metrics['test_r2']:.4f}  "
          f"MAE=${test_metrics['test_mae']:,.0f}  "
          f"RMSE=${test_metrics['test_rmse']:,.0f}  "
          f"CV-test gap={cv_test_gap:+.4f}")

    feature_importances = get_feature_importances(best_model, list(x_train_val.columns))
    if feature_importances:
        print("\nTop feature importances:")
        for feat, score in feature_importances.items():
            print(f"  {feat:<25} {score:.4f}")

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    pickle.dump(best_model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train_val.columns),
              open(output_dir / "model_features.json", 'w'))
    json.dump(compute_percentiles(x_train_val),
              open(output_dir / "model_percentiles.json", 'w'), indent=2)
    json.dump({col: round(float(v), 4) for col, v in x_train_val.median().items()},
              open(output_dir / "model_defaults.json", 'w'), indent=2)

    run_timestamp = datetime.datetime.now(datetime.timezone.utc)
    run_name = f"training_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_experiment("sound-realty-price-prediction")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", "training")
        mlflow.set_tag("mode", mode)
        mlflow.log_param("selected_model", selected_name)
        mlflow.log_param("mode", mode)
        mlflow.log_param("train_val_test_split_ratio", 0.70)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", len(x_train_val.columns))

        for r in cv_results:
            prefix = r["name"].lower().replace(" ", "_")
            mlflow.log_metric(f"{prefix}_cv_mean_r2", r["cv_mean_r2"])
            mlflow.log_metric(f"{prefix}_cv_std_r2", r["cv_std_r2"])

        mlflow.log_metric("test_r2", test_metrics["test_r2"])
        mlflow.log_metric("test_mae", test_metrics["test_mae"])
        mlflow.log_metric("test_rmse", test_metrics["test_rmse"])
        mlflow.log_metric("cv_test_gap", cv_test_gap)

        if feature_importances:
            for feat, score in feature_importances.items():
                mlflow.log_metric(f"importance_{feat}", score)

        mlflow.log_artifact(str(output_dir / "model.pkl"))
        mlflow.log_artifact(str(output_dir / "model_features.json"))
        mlflow.log_artifact(str(output_dir / "model_percentiles.json"))

        run_id = run.info.run_id

    json.dump(
        {"mlflow_run_id": run_id, "mlflow_run_name": run_name,
         "model_type": selected_name, "timestamp": run_timestamp.isoformat()},
        open(output_dir / "model_metadata.json", "w"), indent=2)

    results_dir = pathlib.Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "mlflow_run_id": run_id,
        "mlflow_run_name": run_name,
        "run_type": "training",
        "mode": mode,
        "timestamp": run_timestamp.isoformat(),
        "selected_model": selected_name,
        "train_val_test_split_ratio": 0.70,
        "random_state": 42,
        "n_features": len(x_train_val.columns),
        "feature_names": list(x_train_val.columns),
        "models": models_list,
        "final_test_metrics": {**test_metrics, "cv_test_gap": cv_test_gap},
        "feature_importances_top10": feature_importances,
    }
    json.dump(snapshot, open(results_dir / f"{run_name}.json", "w"), indent=2)

    print(f"\nArtifacts saved to {output_dir}/")
    print(f"Run snapshot saved to {results_dir}/{run_name}.json")
    print(f"MLflow run ID: {run_id}  (run 'mlflow ui' to explore)")


if __name__ == "__main__":
    main()
