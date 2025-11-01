#!/usr/bin/env python3
"""Scaffold generator: writes MCP Flask Middleware files into current directory."""
from pathlib import Path
content_map = {}

content_map = {
    'app.py': 'from flask import Flask, request, jsonify\nfrom client import call_hf_inference\nimport os\n\napp = Flask(__name__)\n\n@app.route("/health", methods=["GET"])\ndef health():\n    return jsonify({"status": "ok"}), 200\n\n@app.route("/infer", methods=["POST"])\ndef infer():\n    # Accept JSON payload and forward to HF endpoint via client\n    payload = request.get_json(force=True)\n    if not payload:\n        return jsonify({"error": "Empty payload"}), 400\n\n    try:\n        response = call_hf_inference(payload)\n    except Exception as e:\n        app.logger.exception("Inference call failed")\n        return jsonify({"error": str(e)}), 502\n\n    return jsonify({"result": response}), 200\n\nif __name__ == \'__main__\':\n    port = int(os.environ.get("MCP_PORT", 8080))\n    app.run(host=\'0.0.0.0\', port=port)\n',
    'client.py': 'import os\nimport requests\nfrom time import sleep\n\nHF_ENDPOINT = os.environ.get("HF_ENDPOINT_URL")\nHF_API_KEY = os.environ.get("HF_API_KEY")\nTIMEOUT = int(os.environ.get("HF_TIMEOUT_S", 30))\nMAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", 3))\nBACKOFF_FACTOR = float(os.environ.get("HF_BACKOFF_FACTOR", 1.5))\n\nif not HF_ENDPOINT or not HF_API_KEY:\n    raise RuntimeError("HF_ENDPOINT_URL and HF_API_KEY must be set in environment")\n\nHEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Accept": "application/json"}\n\n\ndef call_hf_inference(payload: dict) -> dict:\n    \\"\\"\\"Call the HF Inference Endpoint with simple retries and backoff.\n\n    Args:\n        payload: JSON-serializable dict to send in the POST body.\n\n    Returns:\n        Parsed JSON response from HF.\n    \\"\\"\\"\n    url = HF_ENDPOINT\n    for attempt in range(1, MAX_RETRIES + 1):\n        try:\n            resp = requests.post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)\n            resp.raise_for_status()\n            return resp.json()\n        except requests.RequestException as e:\n            if attempt == MAX_RETRIES:\n                raise\n            sleep_time = BACKOFF_FACTOR ** attempt\n            sleep(sleep_time)\n    raise RuntimeError("Unreachable")\n',
    'requirements.txt': 'flask>=2.2.0\nrequests>=2.28.0\ngunicorn>=20.1.0\npython-dotenv>=1.0.0\n',
    'Dockerfile': '# Use slim Python image\nFROM python:3.11-slim\n\n# Create non-root user\nRUN useradd --create-home --shell /bin/bash appuser\nWORKDIR /app\n\n# Install system deps for requests (if needed)\nRUN apt-get update && apt-get install -y --no-install-recommends \\\n    build-essential \\\n    ca-certificates \\\n  && rm -rf /var/lib/apt/lists/*\n\n# Copy requirements first for faster rebuilds\nCOPY requirements.txt /app/requirements.txt\nRUN pip install --no-cache-dir -r /app/requirements.txt\n\n# Copy app\nCOPY . /app\nRUN chown -R appuser:appuser /app\nUSER appuser\n\nEXPOSE 8080\nENV PYTHONUNBUFFERED=1\n\nENTRYPOINT ["/app/entrypoint.sh"]\n',
    'entrypoint.sh': '#!/usr/bin/env bash\nset -e\n\n# load .env if present (only in dev)\nif [ -f ".env" ]; then\n  export $(grep -v \'^#\' .env | xargs)\nfi\n\n# Default gunicorn command\nexec gunicorn -c /app/gunicorn_conf.py app:app\n',
    'gunicorn_conf.py': 'import multiprocessing\n\nbind = "0.0.0.0:8080"\nworkers = int((multiprocessing.cpu_count() * 2) + 1)\nworker_class = "gthread"\nthreads = 4\ntimeout = 120\naccesslog = \'-\'\nerrorlog = \'-\'\n',
    '.env.example': '# Example environment variables — copy to .env and fill in\nHF_ENDPOINT_URL=https://api-inference.huggingface.co/models/your-username/your-model\nHF_API_KEY=hf_xxxREPLACE_WITH_YOUR_TOKENxxx\nMCP_PORT=8080\nHF_TIMEOUT_S=30\nHF_MAX_RETRIES=3\nHF_BACKOFF_FACTOR=1.5\n',
    'docker-compose.yml': 'version: \'3.8\'\nservices:\n  mcp:\n    build: .\n    container_name: mcp_service\n    restart: unless-stopped\n    ports:\n      - "8080:8080"\n    env_file: .env\n    environment:\n      - MCP_PORT=8080\n    healthcheck:\n      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]\n      interval: 30s\n      timeout: 10s\n      retries: 3\n',
    'integration_examples/call_hf_endpoint.py': '\\"\\"\\"Standalone example that calls the HF model endpoint directly (for testing).\\"\\"\\"\nimport os\nimport requests\n\nurl = os.environ.get(\'HF_ENDPOINT_URL\')\napi_key = os.environ.get(\'HF_API_KEY\')\n\nheaders = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}\n\npayload = {"inputs": "Translate to French: I love automation."}\nresp = requests.post(url, json=payload, headers=headers, timeout=30)\nprint(resp.status_code)\nprint(resp.text)\n',
    'integration_examples/call_from_deepcompliance.py': '\\"\\"\\"Example snippet showing how your DeepComplianceAgent can call MCP.\\"\\"\\"\nimport requests\nimport os\n\nMCP_URL = os.environ.get(\'MCP_URL\', \'http://localhost:8080\')\n\npayload = {\n    "inputs": "Summarize the following: <your text here>",\n}\nresp = requests.post(f"{MCP_URL}/infer", json=payload, timeout=60)\nprint(resp.status_code)\nprint(resp.json())\n',
    'README.md': "# MCP Flask Middleware (Docker-ready)\n\nThis repo contains a small Flask microservice (MCP) that forwards inference requests to a Hugging Face Inference Endpoint. It's Docker-ready and intended to act as an integration layer for DeepComplianceAgent.\n\n## Quick start (local, dev)\n\n1. Copy `.env.example` to `.env` and fill `HF_ENDPOINT_URL` and `HF_API_KEY`.\n2. Build & run with docker-compose:\n\n```bash\nchmod +x entrypoint.sh\ndocker-compose up --build\n```\n\n3. Health: `GET http://localhost:8080/health`\n4. Inference: `POST http://localhost:8080/infer` with JSON body (same payload you would send to HF).\n\n## Deploying to production\n- Use container registry (Docker Hub, ECR) and run on your cloud provider (ECS, ECS/Fargate, VM, k8s).\n- Use a secrets manager rather than `.env` for HF API keys.\n- Add TLS termination/load balancer in front of the MCP service.\n\n## Environment variables\nSee `.env.example`.\n\n## Security & ops notes\n- Do not commit `.env` to source control.\n- Limit outgoing concurrency to avoid HF overage costs.\n- Add authentication to `/infer` (API key, mTLS) if used in production.\n",
}


def write_all(base_path='.'):
    base = Path(base_path)
    for p, c in content_map.items():
        fp = base / p
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(c)
    # make entrypoint executable
    import os, stat
    ep = Path(base) / 'entrypoint.sh'
    if ep.exists():
        os.chmod(ep, os.stat(ep).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

if __name__ == '__main__':
    write_all('.')
    print('Scaffold written to current directory.')

