"""
Simple demo client for the Sound Realty prediction API.
Reads 5 examples from data/future_unseen_examples.csv and submits each
as a POST request to /predict, then prints the result.

Usage:
    python test_api.py [--url http://localhost:8000]
"""

import argparse
import sys

import pandas
import requests

DEFAULT_URL = "http://localhost:8000"


def main(base_url):
    """Submit sample house records to the prediction API and print results.

    Args:
        base_url: base URL of the running API service, e.g. http://localhost:8000
    """
    examples = pandas.read_csv("data/future_unseen_examples.csv", dtype={"zipcode": str})
    sample = examples.head(5)

    print(f"Submitting {len(sample)} examples to {base_url}/predict\n")
    print(f"{'#':<4} {'sqft_living':>12} {'bedrooms':>9} {'zipcode':>9} {'predicted_price':>16}")
    print("-" * 55)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        payload = row.to_dict()
        payload["zipcode"] = str(payload["zipcode"]).split(".")[0]

        try:
            response = requests.post(f"{base_url}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            price = result["predicted_price"]
            version = result.get("model_version", "?")
            print(
                f"{i:<4} {int(row['sqft_living']):>12,} {int(row['bedrooms']):>9} "
                f"{payload['zipcode']:>9} ${price:>15,.0f}  [model {version}]"
            )
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Could not connect to {base_url}. Is the service running?")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"ERROR on row {i}: {e} — {response.text}")

    print("\nDone.")


def test_basic(base_url):
    """Submit one request to /predict/basic using only the 8 required fields.

    Args:
        base_url: base URL of the running API service, e.g. http://localhost:8000
    """
    examples = pandas.read_csv("data/future_unseen_examples.csv", dtype={"zipcode": str})
    row = examples.iloc[0]

    payload = {
        "bedrooms": float(row["bedrooms"]),
        "bathrooms": float(row["bathrooms"]),
        "sqft_living": float(row["sqft_living"]),
        "sqft_lot": float(row["sqft_lot"]),
        "floors": float(row["floors"]),
        "sqft_above": float(row["sqft_above"]),
        "sqft_basement": float(row["sqft_basement"]),
        "zipcode": str(row["zipcode"]).split(".")[0],
    }

    print(f"\nSubmitting 1 example to {base_url}/predict/basic (8 fields only)\n")
    print(f"{'sqft_living':>12} {'bedrooms':>9} {'zipcode':>9} {'predicted_price':>16}")
    print("-" * 50)

    try:
        response = requests.post(f"{base_url}/predict/basic", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        price = result["predicted_price"]
        version = result.get("model_version", "?")
        warnings = result.get("warnings", [])
        print(
            f"{int(row['sqft_living']):>12,} {int(row['bedrooms']):>9} "
            f"{payload['zipcode']:>9} ${price:>15,.0f}  [model {version}]"
        )
        if warnings:
            print("\nImputed fields:")
            for w in warnings:
                print(f"  - {w}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {base_url}. Is the service running?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: {e} — {response.text}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Sound Realty prediction API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the API service")
    parser.add_argument("--basic", action="store_true", help="Test the /predict/basic endpoint")
    args = parser.parse_args()
    if args.basic:
        test_basic(args.url)
    else:
        main(args.url)
