"""Standalone example that calls the HF model endpoint directly (for testing)."""
import os
import requests

url = os.environ.get('HF_ENDPOINT_URL')
api_key = os.environ.get('HF_API_KEY')

headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

payload = {"inputs": "Translate to French: I love automation."}
resp = requests.post(url, json=payload, headers=headers, timeout=30)
print(resp.status_code)
print(resp.text)
