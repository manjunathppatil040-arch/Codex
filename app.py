"""
Stock Price Direction Predictor - Flask API
===========================================
REST API to serve predictions from the trained model.

Endpoints:
  POST /api/predict        { "ticker": "AAPL" }
  POST /api/train          { "ticker": "AAPL" }
  GET  /api/health         health check
  GET  /api/tickers        list of popular tickers
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from train_model import train, predict, CONFIG

app = Flask(__name__)
CORS(app)

POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_ready": os.path.exists(CONFIG["model_path"])})


@app.route("/api/tickers", methods=["GET"])
def tickers():
    return jsonify({"tickers": POPULAR_TICKERS})


@app.route("/api/train", methods=["POST"])
def train_endpoint():
    body   = request.get_json(silent=True) or {}
    ticker = body.get("ticker", CONFIG["ticker"]).upper().strip()
    try:
        _, _, metrics = train(ticker)
        return jsonify({"success": True, "ticker": ticker, "metrics": metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    body   = request.get_json(silent=True) or {}
    ticker = body.get("ticker", CONFIG["ticker"]).upper().strip()
    if not os.path.exists(CONFIG["model_path"]):
        return jsonify({"error": "Model not trained yet. Call /api/train first."}), 400
    try:
        result = predict(ticker)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
