"""Painel Streamlit para inspecionar os artefatos de treinamento dos modelos UNSW-NB15."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Painel de Treinamento UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
)

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Load aggregate model comparison metrics."""
    path = BASE_DIR / "model_comparison_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df.sort_values("weighted_f1", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_per_class_f1() -> pd.DataFrame:
    """Load per-class F1 scores for each model."""
    path = BASE_DIR / "per_class_f1_long.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dataset(name: str, sample_size: int) -> pd.DataFrame:
    """Load a parquet dataset and optionally down-sample for responsiveness."""
    path = DATASETS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df


@st.cache_data(show_spinner=False)
def load_confusion_matrix_files() -> list[Path]:
    """Return available confusion matrix images."""
    return sorted(BASE_DIR.glob("confusion_matrix_*.png"))


@st.cache_data(show_spinner=False)
def load_summary_images() -> list[Path]:
    """Return aggregate summary plots, if they exist."""
    files = ["macro_f1_models.png", "weighted_f1_models.png"]
    return [BASE_DIR / fname for fname in files if (BASE_DIR / fname).exists()]


@st.cache_resource(show_spinner=False)
def load_best_model():
    """Load the best-performing pipeline saved during training."""
    model_files = sorted(BASE_DIR.glob("best_model_pipeline_*.joblib"))
    if not model_files:
        return None, None
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    pipeline = joblib.load(latest)
    model_name = latest.stem.replace("best_model_pipeline_", "")
    return pipeline, model_name


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    path = BASE_DIR / "label_encoder.joblib"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource(show_spinner=False)
def load_model_columns():
    path = BASE_DIR / "model_columns.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def preprocess_raw(df_raw: pd.DataFrame, expected_columns: pd.Index):
    """Mirror the training pre-processing to align features for inference."""
    if df_raw.empty:
        return pd.DataFrame(), pd.Series(dtype="object")

    df_filtered = df_raw[
        df_raw["attack_cat"].isin(["Worms", "Backdoor"]) | (df_raw["label"] == 0)
    ].copy()
    if df_filtered.empty:
        return pd.DataFrame(), pd.Series(dtype="object")

    df_filtered["attack_label"] = df_filtered["attack_cat"].fillna("Normal")
    drop_candidates = ["id", "label", "attack_cat"]
    features = df_filtered.drop(columns=[c for c in drop_candidates if c in df_filtered.columns])

    y = features.pop("attack_label")
    X = pd.get_dummies(features)
    if expected_columns is not None:
        X = X.reindex(columns=expected_columns, fill_value=0)
    return X, y


def render_overview(metrics_df: pd.DataFrame, model_name: str | None) -> None:
    """Renderiza a visão geral do painel com métricas de alto nível."""
    st.title("Painel de Treinamento UNSW-NB15")
    st.caption(
        "Visualize os resultados do treinamento, métricas e artefatos diagnósticos dos modelos de detecção de anomalias."
    )

    if model_name:
        st.info(f"Pipeline campeão salvo do treinamento: **{model_name}**")

    if metrics_df.empty:
        st.warning("`model_comparison_metrics.csv` não encontrado. Execute o script de treinamento para gerá-lo.")
        return

    best_row = metrics_df.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Melhor modelo", best_row["model"])
    col2.metric("F1 Ponderado", f"{best_row['weighted_f1']:.4f}")
    col3.metric("F1 Macro", f"{best_row['macro_f1']:.4f}")

    st.markdown("### Comparação entre modelos")
    formatted = metrics_df.style.format(
        {col: "{:.4f}" for col in metrics_df.columns if col != "model"}
    )
    st.dataframe(formatted, use_container_width=True)

    with st.expander("Gráficos-resumo", expanded=False):
        summary_images = load_summary_images()
        if summary_images:
            cols = st.columns(len(summary_images))
            for col, path in zip(cols, summary_images):
                col.image(str(path), caption=path.stem.replace("_", " ").title(), use_column_width=True)
        else:
            st.info(
                "Nenhum gráfico agregado encontrado. Adicione `macro_f1_models.png` ou `weighted_f1_models.png` para exibi-los."
            )


def render_metrics(metrics_df: pd.DataFrame, per_class_df: pd.DataFrame) -> None:
    """Renderiza visualizações detalhadas das métricas."""
    st.header("Análise detalhada de métricas")

    if metrics_df.empty:
        st.info("A tabela de métricas está vazia. Execute `model_training.py` para gerar `model_comparison_metrics.csv`.")
    else:
        st.subheader("F1 ponderado por modelo")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=metrics_df, x="model", y="weighted_f1", palette="viridis", ax=ax)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("F1 ponderado")
        ax.set_xlabel("")
        ax.bar_label(ax.containers[0], fmt="%.3f")
        st.pyplot(fig, use_container_width=True)

        st.subheader("F1 macro por modelo")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=metrics_df, x="model", y="macro_f1", palette="mako", ax=ax)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("F1 macro")
        ax.set_xlabel("")
        ax.bar_label(ax.containers[0], fmt="%.3f")
        st.pyplot(fig, use_container_width=True)

    if per_class_df.empty:
        st.info("Tabela de F1 por classe não encontrada (`per_class_f1_long.csv`).")
    else:
        st.subheader("Comparação de F1 por classe")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=per_class_df, x="class", y="f1", hue="model", palette="Set2", ax=ax)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("F1")
        ax.set_xlabel("")
        ax.legend(title="Modelo")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", label_type="edge", padding=2, fontsize=8)
        st.pyplot(fig, use_container_width=True)


def render_confusion_matrices() -> None:
    """Exibe as matrizes de confusão salvas."""
    st.header("Matrizes de confusão")
    files = load_confusion_matrix_files()
    if not files:
        st.info("Imagens de matrizes de confusão não encontradas. Execute `model_training.py` para gerá-las novamente.")
        return

    cols = st.columns(min(3, len(files)))
    for idx, path in enumerate(files):
        cols[idx % len(cols)].image(
            str(path), caption=path.stem.replace("_", " ").title(), use_column_width=True
        )


def render_dataset_explorer() -> None:
    """Exploração interativa dos datasets de treino e teste."""
    st.header("Explorador de dados")

    sample_size = st.slider(
        "Tamanho da amostra por conjunto",
        min_value=500,
        max_value=5000,
        step=500,
        value=2000,
        help="A amostragem mantém o painel responsivo mesmo com arquivos volumosos.",
    )

    train_df = load_dataset("UNSW_NB15_training-set.parquet", sample_size)
    test_df = load_dataset("UNSW_NB15_testing-set.parquet", sample_size)

    if train_df.empty or test_df.empty:
        st.info(
            "Não foi possível carregar os arquivos parquet em `datasets/`. Verifique se os conjuntos de treino e teste estão presentes."
        )
        return

    st.subheader("Distribuição de classes (amostra de treino)")
    class_counts = train_df["attack_cat"].fillna("Normal").value_counts().sort_index()
    st.bar_chart(class_counts)

    st.subheader("Principais valores categóricos")
    protos = train_df["proto"].value_counts().head(10)
    states = train_df["state"].value_counts().head(10)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Frequência de protocolos (top 10)")
        st.table(protos)
    with col2:
        st.write("Frequência de estados (top 10)")
        st.table(states)

    st.subheader("Resumo de variáveis numéricas")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(train_df[numeric_cols].describe().transpose(), use_container_width=True)
    else:
        st.info("Nenhuma coluna numérica encontrada na amostra.")

    with st.expander("Amostras brutas (treino)"):
        st.dataframe(train_df.head(100), use_container_width=True)
    with st.expander("Amostras brutas (teste)"):
        st.dataframe(test_df.head(100), use_container_width=True)


def render_inference_sandbox(
    model, model_name: str | None, encoder, expected_columns: pd.Index | None
) -> None:
    """Permite executar o modelo salvo em linhas amostradas."""
    st.header("Sandbox de inferência")

    if model is None or encoder is None or expected_columns is None:
        st.info(
            "Os artefatos do modelo (`best_model_pipeline_*.joblib`, `label_encoder.joblib`, `model_columns.joblib`) são necessários para realizar inferência."
        )
        return

    st.write(
        "Escolha uma linha do conjunto de teste para comparar o rótulo previsto com o rótulo real do pipeline salvo."
    )
    if model_name:
        st.caption(f"Pipeline carregado: **{model_name}**")

    base_df = load_dataset("UNSW_NB15_testing-set.parquet", sample_size=5000)
    if base_df.empty:
        st.info("Dataset de teste indisponível para amostragem.")
        return

    X_processed, y_labels = preprocess_raw(base_df, expected_columns)
    if X_processed.empty:
        st.warning(
            "Nenhuma linha correspondeu às classes filtradas (Normal, Worms, Backdoor) na amostra atual."
        )
        return

    row_indices = X_processed.index.tolist()
    format_func = lambda idx: f"Amostra {idx} • Rótulo real: {y_labels.loc[idx]}"
    selected_idx = st.selectbox("Escolha uma amostra", row_indices, format_func=format_func)

    sample_features = X_processed.loc[[selected_idx]]
    actual_label = y_labels.loc[selected_idx]

    pred_encoded = model.predict(sample_features)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]
    probabilities = model.predict_proba(sample_features)[0]

    probability_map = dict(zip(encoder.classes_, probabilities))
    confidence = probability_map.get(pred_label, float("nan"))

    col1, col2 = st.columns(2)
    col1.metric("Rótulo previsto", pred_label)
    col1.metric("Rótulo real", actual_label)
    if np.isnan(confidence):
        col2.metric("Confiança", "N/A")
    else:
        col2.metric("Confiança", f"{confidence:.3f}")

    st.subheader("Distribuição de probabilidades")
    prob_df = (
        pd.DataFrame({"Classe": encoder.classes_, "Probabilidade": probabilities})
        .sort_values(by="Probabilidade", ascending=False)
        .set_index("Classe")
    )
    st.bar_chart(prob_df)

    with st.expander("Valores das features"):
        st.json(base_df.loc[selected_idx].to_dict())


def main() -> None:
    metrics_df = load_metrics()
    per_class_df = load_per_class_f1()
    model, model_name = load_best_model()
    encoder = load_label_encoder()
    expected_columns = load_model_columns()

    nav = st.sidebar.radio(
        "Navegação",
        (
            "Visão Geral",
            "Métricas",
            "Matrizes de Confusão",
            "Explorador de Dados",
            "Sandbox de Inferência",
        ),
        help="Use a navegação para explorar os diferentes artefatos produzidos durante o treinamento.",
    )

    if nav == "Visão Geral":
        render_overview(metrics_df, model_name)
    elif nav == "Métricas":
        render_metrics(metrics_df, per_class_df)
    elif nav == "Matrizes de Confusão":
        render_confusion_matrices()
    elif nav == "Explorador de Dados":
        render_dataset_explorer()
    else:
        render_inference_sandbox(model, model_name, encoder, expected_columns)


if __name__ == "__main__":
    main()
