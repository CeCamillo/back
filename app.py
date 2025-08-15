import io
import uvicorn
import pandas as pd
from joblib import load
from typing import List, Dict, Union, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="API de Predição de Anomalias de Rede",
    description="API para predição de anomalias de rede.",
    version="2.0.0"
)


# --- Model and Feature Loading ---
model = load('model.joblib')
scaler = load('scaler.joblib')
model_columns = load('model_columns.joblib')

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
    prediction: str = Field(..., description="A label da classe predita (exemplo: 'Normal', 'Backdoor').")
    is_anomaly: bool = Field(..., description="True se a predição não for 'Normal'.")
    confidence: float = Field(..., description="A confiança do modelo na sua predição (probabilidade da classe predita).")
    probabilities: Dict[str, float] = Field(..., description="Um dicionário de probabilidades para cada classe possível.")

class EnhancedPredictionResponse(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="Uma cópia dos dados de entrada originais para esta predição.")
    result: PredictionResult = Field(..., description="Os resultados da predição.")


# --- Core Prediction Logic ---

def process_and_predict(df: pd.DataFrame, original_data: List[Dict]) -> List[EnhancedPredictionResponse]:
    """
    Executa o pipeline completo de predição e formata a saída para a API.
    """
    if df.empty:
        raise ValueError("Os dados de entrada não podem estar vazios.")

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
        print(predictions)
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
    Fornece as N features mais importantes como determinadas pelo modelo durante o treinamento.
    Isso é útil para criar visualizações front-end sobre o que o modelo considera
    mais influente globalmente.
    """
    return top_features


@app.post("/predict", response_model=Union[EnhancedPredictionResponse, List[EnhancedPredictionResponse]])
def predict(data: Union[List[Dict], Dict]):
    """
    Endpoint de predição para dados JSON.
    Aceita um único objeto JSON ou uma lista de objetos JSON e retorna uma resposta enriquecida.
    """
    try:
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
            processed_results = process_and_predict(input_df, [data])
            return processed_results[0]
        elif isinstance(data, list):
            if not data:
                raise HTTPException(status_code=400, detail="A lista de entrada não pode estar vazia.")
            input_df = pd.DataFrame(data)
            return process_and_predict(input_df, data)
        else:
            raise HTTPException(status_code=400, detail="A entrada deve ser um objeto JSON ou uma lista de objetos JSON.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado: {e}")


@app.post("/predict/csv", response_model=List[EnhancedPredictionResponse])
async def predict_csv(file: UploadFile = File(...)):
    """
    Endpoint de predição para upload de arquivos CSV.
    Retorna uma resposta de predição enriquecida para cada linha no CSV.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Por favor, envie um arquivo CSV.")
    
    try:
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        input_df = pd.read_csv(buffer)
        
        original_data = input_df.to_dict(orient='records')
        
        return process_and_predict(input_df, original_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado ao processar o arquivo: {e}")


@app.post("/predict/parquet", response_model=List[EnhancedPredictionResponse])
async def predict_parquet(file: UploadFile = File(...)):
    """
    Endpoint de predição para upload de arquivos Parquet.
    Retorna uma resposta de predição enriquecida para cada linha no Parquet.
    """
    if not (file.filename.endswith('.parquet') or file.filename.endswith('.parq')):
        raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Por favor, envie um arquivo Parquet (.parquet).")

    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        input_df = pd.read_parquet(buffer)

        original_data = input_df.to_dict(orient='records')

        return process_and_predict(input_df, original_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado ao processar o arquivo Parquet: {e}")


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True) 