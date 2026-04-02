import json
import logging
import os
import pickle
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "model/model_features.json")
PERCENTILES_PATH = os.getenv("PERCENTILES_PATH", "model/model_percentiles.json")
DEMOGRAPHICS_PATH = os.getenv("DEMOGRAPHICS_PATH", "data/zipcode_demographics.csv")
DEFAULTS_PATH = os.getenv("DEFAULTS_PATH", "model/model_defaults.json")
MODEL_NAME = os.getenv("MODEL_NAME", "sound-realty-price-predictor")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
SCHEMA_VERSION = "1.0"

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(FEATURES_PATH) as f:
    MODEL_FEATURES = json.load(f)

if os.path.exists(DEFAULTS_PATH):
    with open(DEFAULTS_PATH) as f:
        MODEL_DEFAULTS = json.load(f)
else:
    MODEL_DEFAULTS = {}
    logger.warning(f"Defaults file not found at {DEFAULTS_PATH}, /predict/basic median imputation disabled")

if os.path.exists(PERCENTILES_PATH):
    with open(PERCENTILES_PATH) as f:
        MODEL_PERCENTILES = json.load(f)
else:
    MODEL_PERCENTILES = {}
    logger.warning(f"Percentiles file not found at {PERCENTILES_PATH}, out-of-range warnings disabled")

DEMOGRAPHICS = pandas.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})

app = FastAPI(title="Sound Realty Price Predictor")


class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float


class BasicHouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str
    waterfront: Optional[int] = None
    view: Optional[int] = None
    condition: Optional[int] = None
    grade: Optional[int] = None
    yr_built: Optional[int] = None
    yr_renovated: Optional[int] = None
    lat: Optional[float] = None
    long: Optional[float] = None
    sqft_living15: Optional[float] = None
    sqft_lot15: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_price: float
    warnings: list
    data_quality_score: float
    model_name: str
    model_version: str
    model_stage: str
    schema_version: str
    prediction_latency_ms: float
    timestamp: str


def generate_warnings(input_dict: dict, percentiles: dict) -> list:
    """Check input values for cross-field inconsistencies and out-of-range conditions.

    Args:
        input_dict: dict of house feature values provided by the caller,
            with zipcode removed
        percentiles: dict mapping feature names to p5 and p95 bounds computed
            from the training data

    Returns:
        List of warning strings describing any suspicious values found
    """
    found = []

    sqft_diff = abs(
        input_dict["sqft_above"] + input_dict["sqft_basement"] - input_dict["sqft_living"]
    )
    if sqft_diff > 1:
        found.append(
            "sqft_above + sqft_basement does not equal sqft_living, possible data entry error"
        )

    for feature, bounds in percentiles.items():
        if feature not in input_dict:
            continue
        value = input_dict[feature]
        if value < bounds["p5"] or value > bounds["p95"]:
            found.append(
                f"{feature}={value} is outside the typical training range "
                f"(p5={bounds['p5']}, p95={bounds['p95']})"
            )

    return found


@app.get("/health")
def health():
    """Return service health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    """Predict the sale price of a house.

    Args:
        house: validated house features from the request body, corresponding
            to the columns in future_unseen_examples.csv

    Returns:
        PredictionResponse containing the predicted price, input quality
        warnings, traceability metadata, and a UTC timestamp
    """
    prediction_id = str(uuid.uuid4())
    start = time.time()

    logger.info(f"prediction_id={prediction_id} request={house.model_dump()}")

    demo_row = DEMOGRAPHICS[DEMOGRAPHICS["zipcode"] == house.zipcode]
    if demo_row.empty:
        logger.warning(f"prediction_id={prediction_id} zipcode not found: {house.zipcode}")
        raise HTTPException(
            status_code=422,
            detail=f"zipcode '{house.zipcode}' not found in demographics data",
        )

    input_dict = house.model_dump()
    input_dict.pop("zipcode")

    warnings = generate_warnings(input_dict, MODEL_PERCENTILES)
    data_quality_score = round(max(0.0, 1.0 - len(warnings) * 0.1), 1)

    demo_dict = demo_row.drop(columns="zipcode").iloc[0].to_dict()
    row = {**input_dict, **demo_dict}

    feature_frame = pandas.DataFrame([row])[MODEL_FEATURES]

    price = float(MODEL.predict(feature_frame)[0])

    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(
        f"prediction_id={prediction_id} predicted_price={price:.2f} "
        f"warnings={warnings} data_quality_score={data_quality_score} "
        f"latency_ms={latency_ms}"
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        predicted_price=round(price, 2),
        warnings=warnings,
        data_quality_score=data_quality_score,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_stage=MODEL_STAGE,
        schema_version=SCHEMA_VERSION,
        prediction_latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/basic", response_model=PredictionResponse)
def predict_basic(house: BasicHouseFeatures):
    """Predict sale price from a partial feature set.

    Requires only the 8 core house features plus zipcode. Any of the remaining
    10 features may optionally be included; those not provided are imputed from
    the training-set medians stored in model_defaults.json.

    Args:
        house: house features with 8 required fields and 10 optional fields

    Returns:
        PredictionResponse identical in structure to POST /predict
    """
    prediction_id = str(uuid.uuid4())
    start = time.time()

    logger.info(f"prediction_id={prediction_id} endpoint=basic request={house.model_dump(exclude_none=True)}")

    demo_row = DEMOGRAPHICS[DEMOGRAPHICS["zipcode"] == house.zipcode]
    if demo_row.empty:
        logger.warning(f"prediction_id={prediction_id} zipcode not found: {house.zipcode}")
        raise HTTPException(
            status_code=422,
            detail=f"zipcode '{house.zipcode}' not found in demographics data",
        )

    provided = house.model_dump(exclude_none=True)
    provided.pop("zipcode")

    demo_dict = demo_row.drop(columns="zipcode").iloc[0].to_dict()
    row = {**MODEL_DEFAULTS, **provided, **demo_dict}

    feature_frame = pandas.DataFrame([row])[MODEL_FEATURES]

    price = float(MODEL.predict(feature_frame)[0])
    latency_ms = round((time.time() - start) * 1000, 2)

    imputed = [f for f in MODEL_FEATURES if f not in provided and f not in demo_dict]
    warnings = [f"{f} not provided, median value used" for f in imputed]
    data_quality_score = round(max(0.0, 1.0 - len(warnings) * 0.1), 1)

    logger.info(
        f"prediction_id={prediction_id} endpoint=basic predicted_price={price:.2f} "
        f"imputed={imputed} latency_ms={latency_ms}"
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        predicted_price=round(price, 2),
        warnings=warnings,
        data_quality_score=data_quality_score,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_stage=MODEL_STAGE,
        schema_version=SCHEMA_VERSION,
        prediction_latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
