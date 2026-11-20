import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

# These are the exact predictor columns used when training the model.
FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

@app.get("/")
def home():
    return jsonify({
        "message": "Welcome to the SuperKart Sales Prediction API",
        "endpoints": {
            "single_prediction": "POST /v1/predict",
            "batch_prediction": "POST /v1/predictbatch",
            "health": "GET /health"
        }
    })

@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

def validate_features(dataframe):
    missing = [col for col in FEATURE_COLUMNS if col not in dataframe.columns]
    if missing:
        raise ValueError(
            "Input is missing required model features: " + ", ".join(missing)
        )
    # Keep columns in the same order as model training; extra columns are ignored.
    return dataframe[FEATURE_COLUMNS].copy()

@app.post("/v1/predict")
def predict_single():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object containing the model features."}), 400

        input_df = validate_features(pd.DataFrame([payload]))
        prediction = float(model.predict(input_df)[0])

        return jsonify({"Predicted_Product_Store_Sales_Total": prediction})
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Single prediction failed")
        return jsonify({"error": "Prediction failed. Check the server logs."}), 500

@app.post("/v1/predictbatch")
def predict_batch():
    try:
        uploaded_file = request.files.get("file")
        if uploaded_file is None:
            return jsonify({"error": "Upload a CSV file using the form field named 'file'."}), 400

        input_df = pd.read_csv(uploaded_file)
        model_input = validate_features(input_df)
        predictions = model.predict(model_input)

        results = input_df.copy()
        results["Predicted_Product_Store_Sales_Total"] = predictions
        return jsonify(results.to_dict(orient="records"))
    except (ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed. Check the server logs."}), 500

if __name__ == "__main__":
    # For local testing only; use Gunicorn in the Docker container.
    app.run(host="0.0.0.0", port=7860, debug=False)
