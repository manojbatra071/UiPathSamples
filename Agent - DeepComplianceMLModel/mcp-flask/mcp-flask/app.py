from flask import Flask, request, jsonify
from client import call_hf_inference
import os

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/infer", methods=["POST"])
def infer():
    # Accept JSON payload and forward to HF endpoint via client
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    try:
        response = call_hf_inference(payload)
    except Exception as e:
        app.logger.exception("Inference call failed")
        return jsonify({"error": str(e)}), 502

    return jsonify({"result": response}), 200

if __name__ == '__main__':
    port = int(os.environ.get("MCP_PORT", 8080))
    app.run(host='0.0.0.0', port=port)
