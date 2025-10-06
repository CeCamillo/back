"""
Backend API for Network Anomaly Detection
FastAPI server with /predict endpoint
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Network Anomaly Detection API",
    description="API for detecting Worms and Backdoor attacks in network traffic",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Global variables for loaded artifacts
model = None
scaler = None
model_columns = None
label_encoder = None


def load_model_artifacts():
    """Load the trained model artifacts on startup."""
    global model, scaler, model_columns, label_encoder

    try:
        model_path = BASE_DIR / "model.joblib"
        scaler_path = BASE_DIR / "scaler.joblib"
        columns_path = BASE_DIR / "model_columns.joblib"
        encoder_path = BASE_DIR / "label_encoder.joblib"

        if not all([p.exists() for p in [model_path, scaler_path, columns_path, encoder_path]]):
            raise FileNotFoundError("Missing required model artifacts. Run model_training.py first.")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        model_columns = joblib.load(columns_path)
        label_encoder = joblib.load(encoder_path)

        logger.info("✅ Model artifacts loaded successfully")
        logger.info(f"   - Model: {type(model).__name__}")
        logger.info(f"   - Features: {len(model_columns)}")
        logger.info(f"   - Classes: {list(label_encoder.classes_)}")

    except Exception as e:
        logger.error(f"❌ Failed to load model artifacts: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model artifacts when the server starts."""
    load_model_artifacts()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Network Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload CSV/Parquet file for anomaly detection",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "expected_features": len(model_columns) if model_columns is not None else 0,
        "classes": list(label_encoder.classes_) if label_encoder is not None else []
    }


def preprocess_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess uploaded data to match training format.

    Steps:
    1. One-hot encode categorical features
    2. Align columns with training data
    3. Apply scaler
    """
    # One-hot encode categorical features
    X = pd.get_dummies(df)

    # Align with training columns
    X = X.reindex(columns=model_columns, fill_value=0)

    # Apply scaling
    X_scaled = scaler.transform(X)

    return X_scaled


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict anomalies in uploaded network traffic data.

    Args:
        file: CSV or Parquet file containing network traffic data

    Returns:
        JSON response with:
        - total_connections: Total number of connections analyzed
        - normal_connections: Number of normal connections
        - worms_detected: Number of worm attacks detected
        - backdoors_detected: Number of backdoor attacks detected
        - anomaly_rate: Percentage of anomalous traffic
        - precision: Model precision score
        - recall: Model recall score
        - f1_score: Model F1 score
        - predictions: List of predictions for each connection
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server initialization failed.")

    try:
        # Read uploaded file
        contents = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV or Parquet file."
            )

        logger.info(f"📁 Received file: {file.filename} with {len(df)} records")

        # Store original data for response
        original_df = df.copy()

        # Preprocess data
        X_processed = preprocess_uploaded_data(df)

        # Make predictions
        predictions_encoded = model.predict(X_processed)
        predictions = label_encoder.inverse_transform(predictions_encoded)

        # Calculate probabilities (confidence scores)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_processed)
            confidence_scores = probabilities.max(axis=1).tolist()
        else:
            confidence_scores = [1.0] * len(predictions)

        # Count predictions
        total_connections = len(predictions)
        normal_count = int(np.sum(predictions == 'Normal'))
        worms_count = int(np.sum(predictions == 'Worms'))
        backdoor_count = int(np.sum(predictions == 'Backdoor'))
        anomaly_count = worms_count + backdoor_count
        anomaly_rate = (anomaly_count / total_connections * 100) if total_connections > 0 else 0

        # Calculate performance metrics (if ground truth exists)
        precision = None
        recall = None
        f1 = None

        # Check if ground truth columns exist
        ground_truth_cols = ['label', 'attack_cat', 'attack_label']
        ground_truth_col = None

        for col in ground_truth_cols:
            if col in original_df.columns:
                ground_truth_col = col
                break

        if ground_truth_col:
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score

                # Prepare ground truth
                if ground_truth_col == 'label':
                    # Binary label: 0=Normal, 1=Attack
                    y_true_binary = original_df[ground_truth_col].values
                    y_pred_binary = (predictions != 'Normal').astype(int)

                    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                    recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
                    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

                elif ground_truth_col in ['attack_cat', 'attack_label']:
                    # Multi-class label
                    y_true = original_df[ground_truth_col].fillna('Normal').values

                    # Map to same format as predictions
                    y_true_mapped = []
                    for val in y_true:
                        if val in ['Normal', 'Worms', 'Backdoor']:
                            y_true_mapped.append(val)
                        else:
                            y_true_mapped.append('Normal')

                    y_true_mapped = np.array(y_true_mapped)

                    precision = float(precision_score(y_true_mapped, predictions, average='weighted', zero_division=0))
                    recall = float(recall_score(y_true_mapped, predictions, average='weighted', zero_division=0))
                    f1 = float(f1_score(y_true_mapped, predictions, average='weighted', zero_division=0))

                logger.info(f"📊 Performance Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

            except Exception as e:
                logger.warning(f"Could not calculate performance metrics: {e}")

        # Prepare detailed predictions
        detailed_predictions = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            detailed_predictions.append({
                "connection_id": i,
                "prediction": pred,
                "confidence": round(conf, 4)
            })

        # Build response
        response = {
            "status": "success",
            "file_name": file.filename,
            "total_connections": total_connections,
            "normal_connections": normal_count,
            "worms_detected": worms_count,
            "backdoors_detected": backdoor_count,
            "total_anomalies": anomaly_count,
            "anomaly_rate": round(anomaly_rate, 2),
            "metrics": {
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
                "f1_score": round(f1, 4) if f1 is not None else None
            },
            "predictions": detailed_predictions
        }

        logger.info(f"✅ Analysis complete: {anomaly_count}/{total_connections} anomalies detected ({anomaly_rate:.1f}%)")

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Network Anomaly Detection API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
