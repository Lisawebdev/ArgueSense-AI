"""
app.py — the only file in this project that knows about HTTP.

Trains one ArgumentModel instance when the server starts, then exposes it
through three routes:
  GET  /             -> the dashboard (frontend/index.html)
  GET  /api/metrics  -> live cross-validation metrics, as JSON
  POST /api/analyze  -> runs the full pipeline on submitted text, as JSON
"""

from flask import Flask, jsonify, request, send_from_directory

from analyzer import ArgumentModel

app = Flask(__name__, static_folder="../frontend", static_url_path="")

print("Training models (SVM, KNN, SVR, K-Means)...")
model = ArgumentModel()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/metrics")
def metrics():
    return jsonify(model.metrics())


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    n_clusters = data.get("n_clusters", 3)

    if not isinstance(text, str) or len(text.strip()) < 15:
        return jsonify({
            "error": "Please provide at least 15 characters of text to analyze."
        }), 400

    try:
        n_clusters = int(n_clusters)
    except (TypeError, ValueError):
        n_clusters = 3
    n_clusters = max(1, n_clusters)

    results = model.analyze_essay(text, n_clusters=n_clusters)

    if not results:
        return jsonify({
            "error": "No sentences of at least 15 characters were found in that text."
        }), 400

    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=5000)
