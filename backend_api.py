from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Detecção de Anomalias de Rede",
    description="API para detecção de ataques Worms e Backdoor em tráfego de rede",
    version="1.0.0"
)

# Corrigir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

pipeline = None
model_columns = None
label_encoder = None

def load_model_artifacts():
    """Carrega os artefatos do modelo treinado na inicialização."""
    global pipeline, model_columns, label_encoder

    try:
        pipeline_path = BASE_DIR / "pipeline.joblib"
        columns_path = BASE_DIR / "model_columns.joblib"
        encoder_path = BASE_DIR / "label_encoder.joblib"

        if not all([p.exists() for p in [pipeline_path, columns_path, encoder_path]]):
            raise FileNotFoundError("Artefatos do modelo não encontrados. Execute model_training.py primeiro.")

        pipeline = joblib.load(pipeline_path)
        model_columns = joblib.load(columns_path)
        label_encoder = joblib.load(encoder_path)

        logger.info("✅ Artefatos do modelo carregados com sucesso")
        logger.info(f"   - Pipeline: {type(pipeline).__name__}")
        logger.info(f"   - Etapas do Pipeline: {[step[0] for step in pipeline.steps]}")
        logger.info(f"   - Features: {len(model_columns)}")
        logger.info(f"   - Classes: {list(label_encoder.classes_)}")

    except Exception as e:
        logger.error(f"❌ Falha ao carregar artefatos do modelo: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Carrega os artefatos do modelo quando o servidor inicia."""
    load_model_artifacts()


@app.get("/")
async def root():
    """Endpoint de verificação de saúde."""
    return {
        "status": "online",
        "service": "API de Detecção de Anomalias de Rede",
        "version": "1.0.0",
        "endpoints": {
            "/predict/csv": "POST - Enviar arquivo CSV/Parquet para detecção de anomalias",
            "/health": "GET - Verificar status de saúde da API"
        }
    }


@app.get("/health")
async def health_check():
    """Verificação detalhada de saúde com status do modelo."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "pipeline_steps": [step[0] for step in pipeline.steps] if pipeline is not None else [],
        "expected_features": len(model_columns) if model_columns is not None else 0,
        "classes": list(label_encoder.classes_) if label_encoder is not None else []
    }


def preprocess_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processa os dados enviados para corresponder ao formato de treinamento.

    Etapas:
    1. One-hot encode de features categóricas
    2. Alinha colunas com dados de treinamento

    Nota: Scaling é feito automaticamente pelo pipeline
    """
    # Codificar features categóricas com one-hot encoding
    X = pd.get_dummies(df)

    # Alinhar com colunas de treinamento (preencher colunas faltantes com 0)
    X_aligned = X.reindex(columns=model_columns, fill_value=0)

    return X_aligned


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Prediz anomalias em dados de tráfego de rede enviados.

    Args:
        file: Arquivo CSV ou Parquet contendo dados de tráfego de rede

    Returns:
        Resposta JSON com:
        - total_connections: Número total de conexões analisadas
        - normal_connections: Número de conexões normais
        - worms_detected: Número de ataques worm detectados
        - backdoors_detected: Número de ataques backdoor detectados
        - anomaly_rate: Porcentagem de tráfego anômalo
        - precision: Score de precisão do modelo
        - recall: Score de recall do modelo
        - f1_score: Score F1 do modelo
        - predictions: Lista de predições para cada conexão
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline não carregado. Falha na inicialização do servidor.")

    try:
        # Ler arquivo enviado
        contents = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Formato de arquivo não suportado. Por favor, envie um arquivo CSV ou Parquet."
            )

        logger.info(f"📁 Arquivo recebido: {file.filename} com {len(df)} registros")

        # Armazenar dados originais para resposta
        original_df = df.copy()

        # Pré-processar dados (one-hot encoding e alinhamento de colunas)
        X_processed = preprocess_uploaded_data(df)

        # Fazer predições usando o pipeline (lida com scaling internamente)
        predictions_encoded = pipeline.predict(X_processed)
        predictions = label_encoder.inverse_transform(predictions_encoded)

        # Calcular probabilidades (scores de confiança)
        if hasattr(pipeline, 'predict_proba'):
            probabilities = pipeline.predict_proba(X_processed)
            confidence_scores = probabilities.max(axis=1).tolist()
        else:
            confidence_scores = [1.0] * len(predictions)

        # Contar predições
        total_connections = len(predictions)
        normal_count = int(np.sum(predictions == 'Normal'))
        worms_count = int(np.sum(predictions == 'Worms'))
        backdoor_count = int(np.sum(predictions == 'Backdoor'))
        anomaly_count = worms_count + backdoor_count
        anomaly_rate = (anomaly_count / total_connections * 100) if total_connections > 0 else 0

        # Calcular métricas de desempenho (se ground truth existir)
        # Isso é crucial para validação do TCC - retorna classification_report completo
        full_classification_report = None
        precision = None
        recall = None
        f1 = None

        # Verificar se colunas de ground truth existem
        ground_truth_cols = ['label', 'attack_cat', 'attack_label']
        ground_truth_col = None

        for col in ground_truth_cols:
            if col in original_df.columns:
                ground_truth_col = col
                break

        if ground_truth_col:
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

                # Preparar ground truth
                if ground_truth_col == 'label':
                    # Rótulo binário: 0=Normal, 1=Ataque
                    y_true_binary = original_df[ground_truth_col].values
                    y_pred_binary = (predictions != 'Normal').astype(int)

                    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                    recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
                    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

                    # Gerar relatório de classificação binária
                    full_classification_report = classification_report(
                        y_true_binary,
                        y_pred_binary,
                        target_names=['Normal', 'Attack'],
                        output_dict=True,
                        zero_division=0
                    )

                elif ground_truth_col in ['attack_cat', 'attack_label']:
                    # Rótulo multi-classe - ESTE É O CAMINHO PRINCIPAL PARA VALIDAÇÃO DO TCC
                    y_true = original_df[ground_truth_col].fillna('Normal').values

                    # Mapear para o mesmo formato das predições
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

                    # Gerar relatório de classificação multi-classe completo
                    # Esta é a saída chave para avaliação acadêmica
                    full_classification_report = classification_report(
                        y_true_mapped,
                        predictions,
                        target_names=['Backdoor', 'Normal', 'Worms'],
                        output_dict=True,
                        zero_division=0
                    )

                logger.info(f"📊 Métricas de Performance - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                logger.info(f"📊 Relatório de Classificação Completo Gerado (para validação do TCC)")

            except Exception as e:
                logger.warning(f"Não foi possível calcular métricas de performance: {e}")

        # Preparar predições detalhadas
        detailed_predictions = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            detailed_predictions.append({
                "connection_id": i,
                "prediction": pred,
                "confidence": round(conf, 4)
            })

        # Construir resposta
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

        # Adiciona relatório de classificação completo se disponível (para avaliação acadêmica do TCC)
        if full_classification_report is not None:
            response["classification_report"] = full_classification_report
            logger.info("✅ Relatório de classificação completo incluído na resposta para validação do TCC")

        logger.info(f"✅ Análise completa: {anomaly_count}/{total_connections} anomalias detectadas ({anomaly_rate:.1f}%)")

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro durante predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Iniciando API de Detecção de Anomalias de Rede...")
    print("📍 API estará disponível em: http://localhost:8000")
    print("📖 Documentação da API em: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
