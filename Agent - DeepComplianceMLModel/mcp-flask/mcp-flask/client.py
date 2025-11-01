import os
import requests
from time import sleep

HF_ENDPOINT = os.environ.get("HF_ENDPOINT_URL")
HF_API_KEY = os.environ.get("HF_API_KEY")
TIMEOUT = int(os.environ.get("HF_TIMEOUT_S", 30))
MAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", 3))
BACKOFF_FACTOR = float(os.environ.get("HF_BACKOFF_FACTOR", 1.5))

if not HF_ENDPOINT or not HF_API_KEY:
    raise RuntimeError("HF_ENDPOINT_URL and HF_API_KEY must be set in environment")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Accept": "application/json"}


def call_hf_inference(payload: dict) -> dict:
    """Call the HF Inference Endpoint with simple retries and backoff.

    Args:
        payload: JSON-serializable dict to send in the POST body.

    Returns:
        Parsed JSON response from HF.
    """
    url = HF_ENDPOINT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            sleep_time = BACKOFF_FACTOR ** attempt
            sleep(sleep_time)
    raise RuntimeError("Unreachable")
