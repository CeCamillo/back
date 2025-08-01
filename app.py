from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import io

app = Flask(__name__)

model = load('model.joblib')
scaler = load('scaler.joblib')
model_columns = load('model_columns.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for the model.
    Takes a JSON object or an array of JSON objects.
    """
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'Sem dados de entrada'}), 400

        if isinstance(json_data, dict):
            input_df = pd.DataFrame([json_data])
        elif isinstance(json_data, list):
            input_df = pd.DataFrame(json_data)
        else:
            return jsonify({'error': 'Não é um JSON ou uma lista de JSONs'}), 400

        input_df_encoded = pd.get_dummies(input_df)

        input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        input_df_aligned = input_df_aligned[model_columns]

        input_scaled = scaler.transform(input_df_aligned)

        predictions = model.predict(input_scaled)
        prediction_probabilities = model.predict_proba(input_scaled)

        output = []
        for i, prediction in enumerate(predictions):
            probabilities = dict(zip(model.classes_, prediction_probabilities[i]))
            output.append({
                'prediction': prediction,
                'probabilities': probabilities
            })

        if isinstance(json_data, dict):
            return jsonify(output[0])
        else:
            return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/csv', methods=['POST'])
def predict_csv():
    """
    Prediction endpoint for CSV file uploads.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Arquivo não encontrado'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Arquivo não selecionado'}), 400

        if file and file.filename.endswith('.csv'):
            csv_data = file.read().decode('utf-8')
            input_df = pd.read_csv(io.StringIO(csv_data))

            input_df_encoded = pd.get_dummies(input_df)

            input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
            
            input_df_aligned = input_df_aligned[model_columns]

            input_scaled = scaler.transform(input_df_aligned)

            predictions = model.predict(input_scaled)
            prediction_probabilities = model.predict_proba(input_scaled)

            output = []
            for i, prediction in enumerate(predictions):
                probabilities = dict(zip(model.classes_, prediction_probabilities[i]))
                output.append({
                    'prediction': prediction,
                    'probabilities': probabilities
                })

            return jsonify(output)
        else:
            return jsonify({'error': 'Arquivo não é um CSV'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # tira o debug true se n for usar, roda mais rapido
    app.run(debug=True, port=5000) 