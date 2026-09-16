"""Simulates live traffic against the running API, injecting an artificial
Amount-distribution shift partway through to exercise the drift-monitoring
pipeline end to end.

Requires the API to be up and reachable (`make serve` or `docker compose up`).
Run with: python -m monitoring.simulate_traffic
"""
import pandas as pd
import requests

from src import config

API_URL = "http://localhost:8000"
N_NORMAL = 200
N_DRIFTED = 200
DRIFT_AMOUNT_MULTIPLIER = 8.0
DRIFT_AMOUNT_OFFSET = 500.0


def load_sample(n: int, seed: int) -> pd.DataFrame:
    """Samples `n` random transactions (features only) from the raw dataset."""
    df = pd.read_csv(config.DATA_PATH)
    return df.drop(columns=["Class"]).sample(n=n, random_state=seed).reset_index(drop=True)


def to_payload(row: pd.Series) -> dict:
    """Converts a pandas row to plain Python floats so it's JSON-serializable."""
    return {k: float(v) for k, v in row.items()}


def send(transaction: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=transaction, timeout=5)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    normal_batch = load_sample(N_NORMAL, seed=123)
    drifted_batch = load_sample(N_DRIFTED, seed=456)
    drifted_batch["Amount"] = drifted_batch["Amount"] * DRIFT_AMOUNT_MULTIPLIER + DRIFT_AMOUNT_OFFSET

    print(f"Sending {len(normal_batch)} normal transactions to {API_URL}/predict ...")
    for _, row in normal_batch.iterrows():
        send(to_payload(row))

    print(
        f"Sending {len(drifted_batch)} transactions with injected Amount drift "
        f"(x{DRIFT_AMOUNT_MULTIPLIER} + {DRIFT_AMOUNT_OFFSET}) ..."
    )
    for _, row in drifted_batch.iterrows():
        send(to_payload(row))

    print("Done. Run `python -m monitoring.drift_check` (or `make drift`) to detect the drift.")


if __name__ == "__main__":
    main()
