import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from joblib import dump

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn

# --- Reproducibility ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Load Dataset ---
df_train = pd.read_parquet("./datasets/UNSW_NB15_training-set.parquet")
df_test = pd.read_parquet("./datasets/UNSW_NB15_testing-set.parquet")

def preprocess_data(df):
    """Preprocess dataset and return X, y."""
    df_filtered = df[df['attack_cat'].isin(['Worms', 'Backdoor']) | (df['label'] == 0)].copy()
    df_filtered['attack_label'] = df_filtered['attack_cat'].fillna('Normal')
    df_filtered = df_filtered.drop(columns=[c for c in ['id', 'label', 'attack_cat'] if c in df_filtered])
    
    # Features
    X = df_filtered.drop(columns=['attack_label'])
    # Target
    y = df_filtered['attack_label']
    return X, y

X_train_raw, y_train = preprocess_data(df_train)
X_test_raw, y_test = preprocess_data(df_test)

print("--- Distribuição de Classes no Conjunto de Treinamento ---")
print(y_train.value_counts(), "\n" + "-"*40)

# One-hot encoding + alignment
X_train = pd.get_dummies(X_train_raw)
X_test = pd.get_dummies(X_test_raw)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# Label encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
class_labels = le.classes_

# --- Define Models & Params ---
models = {
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced')
}

param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, 15],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8],
    },
    'LightGBM': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, -1],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [31, 50],
    }
}

# --- Training Loop with MLflow ---
best_score = -1
best_model_pipeline = None
best_model_name = ""

mlflow.set_experiment("UNSW_NB15_Classification")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\n{'='*20}\nTreinando {name}...\n{'='*20}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=10,
            cv=3,
            verbose=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring='f1_weighted'
        )

        search.fit(X_train, y_train_encoded)
        best_pipeline_for_model = search.best_estimator_
        y_pred_encoded = best_pipeline_for_model.predict(X_test)

        # --- Metrics ---
        current_score = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
        report = classification_report(y_test_encoded, y_pred_encoded, target_names=class_labels, output_dict=True)

        print(f"\n--- Resultados para {name} ---")
        print("Melhores Parâmetros:", search.best_params_)
        print(f"F1-Score Ponderado no Conjunto de Teste: {current_score:.4f}")

        # --- MLflow Logging ---
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("f1_weighted", current_score)

        # Log classification report metrics
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)

        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test_encoded, y_pred_encoded, normalize="true")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Matriz de Confusão - {name}')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{name}.png")
        mlflow.log_artifact(f"confusion_matrix_{name}.png")

        # Log model itself
        mlflow.sklearn.log_model(best_pipeline_for_model, artifact_path="model")

        # Track best model
        if current_score > best_score:
            best_score = current_score
            best_model_pipeline = best_pipeline_for_model
            best_model_name = name

# --- Final Save ---
print(f"\n{'='*20}\nMelhor Modelo Geral: {best_model_name} (F1 Ponderado: {best_score:.4f})\n{'='*20}")
dump(best_model_pipeline, f'best_model_pipeline_{best_model_name}.joblib')
dump(X_train.columns, 'model_columns.joblib')
dump(le, 'label_encoder.joblib')
print("Artefatos salvos com sucesso.")