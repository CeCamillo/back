import io
import uvicorn
import pandas as pd
from joblib import load
from typing import List, Dict, Union, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Prediction API",
    description="API for making predictions and providing model insights.",
    version="2.0.0"
)

# --- Model and Feature Loading ---
try:
    model = load('model.joblib')
    scaler = load('scaler.joblib')
    model_columns = load('model_columns.joblib')
except FileNotFoundError as e:
    raise RuntimeError(f"Could not load machine learning asset: {e}. Make sure model.joblib, scaler.joblib, and model_columns.joblib are in the root directory.")

# NEW: Calculate and store global feature importances at startup
TOP_N_FEATURES = 15
feature_importances_df = pd.DataFrame(
    {'feature': model_columns, 'importance': model.feature_importances_}
).sort_values('importance', ascending=False)

top_features = feature_importances_df.head(TOP_N_FEATURES).to_dict('records')


# --- Pydantic Models for API Data Structure ---

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class PredictionResult(BaseModel):
    prediction: str = Field(..., description="The predicted class label (e.g., 'Normal', 'Backdoor').")
    is_anomaly: bool = Field(..., description="True if the prediction is not 'Normal'.")
    confidence: float = Field(..., description="The model's confidence in its prediction (probability of the predicted class).")
    probabilities: Dict[str, float] = Field(..., description="A dictionary of probabilities for each possible class.")

class EnhancedPredictionResponse(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="A copy of the original input data for this prediction.")
    result: PredictionResult = Field(..., description="The prediction results.")


# --- Core Prediction Logic ---

def process_and_predict(df: pd.DataFrame, original_data: List[Dict]) -> List[EnhancedPredictionResponse]:
    """
    Runs the full prediction pipeline and formats the output for the API.
    """
    if df.empty:
        raise ValueError("Input data cannot be empty.")

    # Preprocessing
    input_df_encoded = pd.get_dummies(df)
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    input_df_aligned = input_df_aligned[model_columns]
    input_scaled = scaler.transform(input_df_aligned)

    # Prediction
    predictions = model.predict(input_scaled)
    prediction_probabilities = model.predict_proba(input_scaled)

    # Format output
    output = []
    for i, prediction in enumerate(predictions):
        prediction_str = str(prediction.item() if hasattr(prediction, 'item') else prediction)
        probabilities_dict = dict(zip(model.classes_, prediction_probabilities[i]))
        
        # Ensure values are JSON serializable and get confidence
        probabilities_item = {str(k): v.item() if hasattr(v, 'item') else v for k, v in probabilities_dict.items()}
        confidence = probabilities_item.get(prediction_str, 0.0)

        result_obj = PredictionResult(
            prediction=prediction_str,
            is_anomaly=(prediction_str != 'Normal'),
            confidence=confidence,
            probabilities=probabilities_item
        )
        
        output.append(
            EnhancedPredictionResponse(
                input_data=original_data[i],
                result=result_obj
            )
        )
    return output


# --- API Endpoints ---

@app.get("/features/importances", response_model=List[FeatureImportance])
def get_feature_importances():
    """
    Provides the top N most important features as determined by the model during training.
    This is useful for creating frontend visualizations about what the model considers
    most influential overall.
    """
    return top_features


@app.post("/predict", response_model=Union[EnhancedPredictionResponse, List[EnhancedPredictionResponse]])
def predict(data: Union[List[Dict], Dict]):
    """
    Prediction endpoint for JSON data.
    Accepts a single JSON object or a list of JSON objects and returns an enhanced response.
    """
    try:
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
            processed_results = process_and_predict(input_df, [data])
            return processed_results[0]
        elif isinstance(data, list):
            if not data:
                raise HTTPException(status_code=400, detail="Input list cannot be empty.")
            input_df = pd.DataFrame(data)
            return process_and_predict(input_df, data)
        else:
            raise HTTPException(status_code=400, detail="Input must be a JSON object or a list of JSON objects.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/predict/csv", response_model=List[EnhancedPredictionResponse])
async def predict_csv(file: UploadFile = File(...)):
    """
    Prediction endpoint for CSV file uploads.
    Returns an enhanced prediction response for each row in the CSV.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        input_df = pd.read_csv(buffer)
        
        original_data = input_df.to_dict(orient='records')
        
        return process_and_predict(input_df, original_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing the file: {e}")


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True) 