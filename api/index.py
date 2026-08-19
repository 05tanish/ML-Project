# ==============================================================================
# VERCEL SERVERLESS FUNCTION & API ENTRYPOINT
# Phishing Website Detection — Machine Learning Microservice
# ==============================================================================

import json
import os
from http.server import BaseHTTPRequestHandler
import urllib.parse
import joblib
import pandas as pd
import numpy as np

# Load model artifacts
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_phishing_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "phishing_scaler.pkl")

model = None
scaler = None

def get_model():
    global model, scaler
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            pass
    if scaler is None and os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            pass
    return model, scaler

FEATURE_COLUMNS = [
    "URL_Length", "Num_Dots", "Num_Hyphens", "Num_Special_Chars",
    "Num_Subdomains", "Has_IP_Address", "Has_HTTPS", "Domain_Age_Days",
    "Domain_Registration_Length", "Has_Suspicious_Words", "Num_Redirects",
    "External_Link_Ratio", "Image_Link_Ratio", "Form_Count",
    "Password_Field_Present", "Iframe_Count", "Popup_Count",
    "Favicon_External", "Domain_Name_Length", "URL_Entropy"
]

def predict_from_dict(data):
    mdl, _ = get_model()
    if mdl is None:
        return {"error": "Model artifact not loaded"}, 500

    row = {}
    for feat in FEATURE_COLUMNS:
        row[feat] = float(data.get(feat, 0.0))

    df_in = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred = int(mdl.predict(df_in)[0])
    proba = None
    if hasattr(mdl, "predict_proba"):
        p = mdl.predict_proba(df_in)[0]
        proba = {
            "legitimate_probability": round(float(p[0]) * 100, 2),
            "phishing_probability": round(float(p[1]) * 100, 2)
        }

    return {
        "status": "success",
        "prediction": "PHISHING" if pred == 1 else "LEGITIMATE",
        "prediction_code": pred,
        "probabilities": proba,
        "model": "Tuned Random Forest Classifier (100 Trees, max_depth=20)",
        "accuracy": "97.16%",
        "f1_score": "96.50%"
    }, 200


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Website Detection — AI Security API</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: #0b1120; color: #e2e8f0; min-height: 100vh; padding: 2rem 1rem; }
        .container { max-width: 960px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        .header h1 { font-size: 2.2rem; color: #38bdf8; margin-bottom: 0.5rem; }
        .header p { color: #94a3b8; font-size: 1.05rem; }
        .card { background: #1e293b; border-radius: 16px; padding: 2rem; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.3); margin-bottom: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { font-size: 0.8rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.3rem; }
        .form-group input, .form-group select { background: #0f172a; border: 1px solid #475569; color: #fff; padding: 0.6rem 0.8rem; border-radius: 8px; font-size: 0.9rem; }
        .btn-group { display: flex; gap: 0.8rem; margin: 1.5rem 0 1rem; flex-wrap: wrap; }
        button { cursor: pointer; font-weight: 600; border-radius: 8px; padding: 0.75rem 1.5rem; transition: 0.2s; border: none; }
        .btn-primary { background: #0284c7; color: white; flex: 1; min-width: 180px; font-size: 1rem; }
        .btn-primary:hover { background: #0369a1; }
        .btn-preset { background: #334155; color: #e2e8f0; font-size: 0.85rem; padding: 0.5rem 1rem; }
        .btn-preset:hover { background: #475569; }
        .result-box { display: none; margin-top: 1.5rem; padding: 1.5rem; border-radius: 12px; text-align: center; }
        .result-phishing { background: rgba(220, 38, 38, 0.2); border: 2px solid #ef4444; color: #fca5a5; }
        .result-legit { background: rgba(22, 163, 74, 0.2); border: 2px solid #22c55e; color: #86efac; }
        .progress-bar-wrap { background: #0f172a; height: 16px; border-radius: 8px; overflow: hidden; margin: 1rem 0; display: flex; }
        .bar-legit { background: #22c55e; transition: width 0.4s; }
        .bar-phish { background: #ef4444; transition: width 0.4s; }
        .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: bold; background: #0369a1; color: #e0f2fe; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">Vercel Serverless ML Endpoint</span>
            <h1>🔐 Phishing Website Detection AI</h1>
            <p>Tuned Random Forest Classifier • 97.16% Accuracy • 96.50% F1-Score</p>
        </div>

        <div class="card">
            <h3 style="margin-bottom: 1rem; color: #f8fafc;">⚡ Quick Test Presets</h3>
            <div class="btn-group">
                <button class="btn-preset" onclick="loadPreset('legit')">🏦 Verified Banking Portal</button>
                <button class="btn-preset" onclick="loadPreset('phish')">🚨 High-Risk Phishing</button>
                <button class="btn-preset" onclick="loadPreset('redirect')">⚠️ Malicious Redirect Scam</button>
            </div>

            <h3 style="margin: 1.5rem 0 1rem; color: #f8fafc;">🌐 Website Feature Parameters</h3>
            <form id="predForm" onsubmit="submitForm(event)">
                <div class="grid" id="formGrid"></div>
                <div class="btn-group">
                    <button type="submit" class="btn-primary">🔍 Run Live AI Prediction</button>
                </div>
            </form>

            <div id="resultBox" class="result-box">
                <h2 id="resultTitle" style="font-size: 1.6rem; margin-bottom: 0.5rem;"></h2>
                <div class="progress-bar-wrap">
                    <div id="barLegit" class="bar-legit" style="width: 50%;"></div>
                    <div id="barPhish" class="bar-phish" style="width: 50%;"></div>
                </div>
                <p id="resultProb" style="font-size: 1rem; font-weight: 600;"></p>
            </div>
        </div>
    </div>

    <script>
        const features = [
            "URL_Length", "Num_Dots", "Num_Hyphens", "Num_Special_Chars",
            "Num_Subdomains", "Has_IP_Address", "Has_HTTPS", "Domain_Age_Days",
            "Domain_Registration_Length", "Has_Suspicious_Words", "Num_Redirects",
            "External_Link_Ratio", "Image_Link_Ratio", "Form_Count",
            "Password_Field_Present", "Iframe_Count", "Popup_Count",
            "Favicon_External", "Domain_Name_Length", "URL_Entropy"
        ];

        const binaryCols = ["Has_IP_Address", "Has_HTTPS", "Has_Suspicious_Words", "Password_Field_Present", "Favicon_External"];

        const presets = {
            legit: {"URL_Length":24,"Num_Dots":1,"Num_Hyphens":0,"Num_Special_Chars":1,"Num_Subdomains":0,"Has_IP_Address":0,"Has_HTTPS":1,"Domain_Age_Days":3650,"Domain_Registration_Length":730,"Has_Suspicious_Words":0,"Num_Redirects":0,"External_Link_Ratio":0.15,"Image_Link_Ratio":0.20,"Form_Count":1,"Password_Field_Present":1,"Iframe_Count":0,"Popup_Count":0,"Favicon_External":0,"Domain_Name_Length":10,"URL_Entropy":3.12},
            phish: {"URL_Length":115,"Num_Dots":6,"Num_Hyphens":4,"Num_Special_Chars":7,"Num_Subdomains":3,"Has_IP_Address":1,"Has_HTTPS":0,"Domain_Age_Days":12,"Domain_Registration_Length":30,"Has_Suspicious_Words":1,"Num_Redirects":3,"External_Link_Ratio":0.85,"Image_Link_Ratio":0.90,"Form_Count":4,"Password_Field_Present":1,"Iframe_Count":2,"Popup_Count":3,"Favicon_External":1,"Domain_Name_Length":28,"URL_Entropy":4.88},
            redirect: {"URL_Length":88,"Num_Dots":4,"Num_Hyphens":3,"Num_Special_Chars":5,"Num_Subdomains":2,"Has_IP_Address":0,"Has_HTTPS":0,"Domain_Age_Days":45,"Domain_Registration_Length":90,"Has_Suspicious_Words":1,"Num_Redirects":4,"External_Link_Ratio":0.70,"Image_Link_Ratio":0.60,"Form_Count":3,"Password_Field_Present":0,"Iframe_Count":3,"Popup_Count":4,"Favicon_External":1,"Domain_Name_Length":22,"URL_Entropy":4.52}
        };

        const grid = document.getElementById("formGrid");
        features.forEach(f => {
            const grp = document.createElement("div");
            grp.className = "form-group";
            if (binaryCols.includes(f)) {
                grp.innerHTML = `<label>${f}</label><select id="${f}"><option value="0">0 (No)</option><option value="1">1 (Yes)</option></select>`;
            } else {
                grp.innerHTML = `<label>${f}</label><input type="number" step="any" id="${f}" value="0">`;
            }
            grid.appendChild(grp);
        });

        function loadPreset(name) {
            const p = presets[name];
            for (let k in p) {
                const el = document.getElementById(k);
                if (el) el.value = p[k];
            }
            document.getElementById("predForm").dispatchEvent(new Event("submit"));
        }

        async function submitForm(e) {
            e.preventDefault();
            const data = {};
            features.forEach(f => {
                data[f] = parseFloat(document.getElementById(f).value) || 0;
            });

            try {
                const res = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const json = await res.json();
                const box = document.getElementById("resultBox");
                box.style.display = "block";

                if (json.prediction === "PHISHING") {
                    box.className = "result-box result-phishing";
                    document.getElementById("resultTitle").innerText = "🚨 DANGER: PHISHING WEBSITE DETECTED";
                } else {
                    box.className = "result-box result-legit";
                    document.getElementById("resultTitle").innerText = "✅ SAFE: LEGITIMATE WEBSITE";
                }

                const pPhish = json.probabilities.phishing_probability;
                const pLegit = json.probabilities.legitimate_probability;
                document.getElementById("barLegit").style.width = pLegit + "%";
                document.getElementById("barPhish").style.width = pPhish + "%";
                document.getElementById("resultProb").innerText = `Legitimate: ${pLegit}% | Phishing: ${pPhish}%`;
            } catch(err) {
                alert("Prediction request failed: " + err);
            }
        }

        loadPreset('legit');
    </script>
</body>
</html>
"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ["", "/index", "/api"]:
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif path in ["/health", "/api/health", "/api/status"]:
            mdl, _ = get_model()
            res = {
                "status": "healthy" if mdl is not None else "degraded",
                "model_loaded": mdl is not None,
                "framework": "Scikit-Learn Random Forest",
                "accuracy": "97.16%",
                "f1_score": "96.50%"
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        result, status_code = predict_from_dict(body)
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

# Export WSGI application callable for WSGI / ASGI runners
def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    if method == "POST":
        try:
            request_body_size = int(environ.get("CONTENT_LENGTH", 0))
        except (ValueError):
            request_body_size = 0
        request_body = environ["wsgi.input"].read(request_body_size)
        try:
            data = json.loads(request_body.decode("utf-8"))
        except Exception:
            data = {}
        result, code = predict_from_dict(data)
        response_body = json.dumps(result).encode("utf-8")
        status = f"{code} OK" if code == 200 else f"{code} Error"
        response_headers = [("Content-Type", "application/json"), ("Access-Control-Allow-Origin", "*")]
        start_response(status, response_headers)
        return [response_body]

    if path in ["/health", "/api/health"]:
        mdl, _ = get_model()
        res = {"status": "healthy" if mdl is not None else "degraded", "model_loaded": mdl is not None}
        response_body = json.dumps(res).encode("utf-8")
        start_response("200 OK", [("Content-Type", "application/json")])
        return [response_body]

    response_body = HTML_PAGE.encode("utf-8")
    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
    return [response_body]

app = application
