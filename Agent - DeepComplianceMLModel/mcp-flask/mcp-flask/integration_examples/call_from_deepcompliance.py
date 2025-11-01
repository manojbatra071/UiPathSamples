"""Example snippet showing how your DeepComplianceAgent can call MCP."""
import requests
import os

MCP_URL = os.environ.get('MCP_URL', 'http://localhost:8080')

payload = {
    "inputs": "Summarize the following: <your text here>",
}
resp = requests.post(f"{MCP_URL}/infer", json=payload, timeout=60)
print(resp.status_code)
print(resp.json())
