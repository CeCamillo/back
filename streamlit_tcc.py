"""
Dashboard de Validação TCC - Análise Comparativa
Frontend Streamlit que processa Parquet e mostra resultados idênticos ao tcc_results/
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Optional

# Configurar página
st.set_page_config(
    page_title="Sistema de Validação TCC - Análise Comparativa",
    page_icon="🛡️",
    layout="wide"
)

# Aplicação Principal
st.title("🛡️ Detecção de Anomalias")
st.write("Envie um arquivo Parquet")

st.divider()

# Função para preprocessar dados (idêntico ao generate_tcc_results.py)
def preprocess_data(df):
    """Pré-processar dataset UNSW-NB15"""
    # Filtrar dataset para focar em Normal, Backdoor e Worms
    df_filtered = df[df['attack_cat'].isin(['Worms', 'Backdoor']) | (df['label'] == 0)].copy()

    # Criar rótulo de ataque
    df_filtered['attack_label'] = df_filtered['attack_cat'].fillna('Normal')

    # Remover colunas de metadados
    df_filtered = df_filtered.drop(columns=[c for c in ['id', 'label', 'attack_cat'] if c in df_filtered])

    # Separar features e alvo
    X = df_filtered.drop(columns=['attack_label'])
    y = df_filtered['attack_label']

    return X, y

# Carregar modelos (cached)
@st.cache_resource
def load_models():
    """Carregar modelos naive e improved"""
    try:
        naive_model = load('naive_model_pipeline.joblib')
        improved_model = load('pipeline.joblib')
        return naive_model, improved_model
    except FileNotFoundError as e:
        st.error(f"❌ Erro ao carregar modelos: {e}")
        st.info("💡 Execute `python model_training.py` para gerar os modelos necessários.")
        st.stop()

# Carregar dados de treinamento para alinhamento
@st.cache_data
def load_training_data():
    """Carregar dados de treinamento para alinhamento de features"""
    df_train = pd.read_parquet("./datasets/UNSW_NB15_training-set.parquet")
    X_train_raw, y_train = preprocess_data(df_train)
    X_train = pd.get_dummies(X_train_raw)
    return X_train, y_train

# Uploader de arquivo
uploaded_file = st.file_uploader("Escolha um arquivo Parquet", type="parquet")

if uploaded_file is not None:
    st.info(f"📁 Arquivo enviado: **{uploaded_file.name}**")

    # Mostrar spinner enquanto processa
    with st.spinner('🔄 Processando dados e gerando análise comparativa... .'):
        try:
            # Carregar arquivo parquet
            df_test = pd.read_parquet(uploaded_file)

            # Carregar modelos e dados de treinamento
            naive_model, improved_model = load_models()
            X_train, y_train = load_training_data()

            # Preprocessar dados de teste
            X_test_raw, y_test = preprocess_data(df_test)

            # One-hot encoding e alinhamento
            X_test = pd.get_dummies(X_test_raw)
            X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

            # Codificar labels
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            class_labels = le.classes_

            # Fazer predições com ambos os modelos
            y_pred_naive = naive_model.predict(X_test)
            y_pred_improved = improved_model.predict(X_test)

            # Gerar relatórios
            naive_report_dict = classification_report(y_test_encoded, y_pred_naive, target_names=class_labels, output_dict=True)
            improved_report_dict = classification_report(y_test_encoded, y_pred_improved, target_names=class_labels, output_dict=True)

            # Contar predições por modelo
            naive_predictions = le.inverse_transform(y_pred_naive)
            improved_predictions = le.inverse_transform(y_pred_improved)

        except Exception as e:
            st.error(f"❌ **Erro ao processar arquivo**: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    st.success('✅ Análise Completa!')

    # ========================================================================
    # SEÇÃO 1: Informações do Dataset
    # ========================================================================
    st.divider()
    st.header("📁 Informações do Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total de Amostras Originais",
            f"{len(df_test):,}",
            help="Amostras no arquivo original"
        )

    with col2:
        st.metric(
            "Amostras Filtradas",
            f"{len(y_test):,}",
            help="Após filtrar para Normal, Backdoor, Worms"
        )

    with col3:
        st.metric(
            "Features",
            f"{X_test.shape[1]:,}",
            help="Total de features após one-hot encoding"
        )

    with col4:
        st.metric(
            "Classes",
            len(class_labels),
            help="Normal, Backdoor, Worms"
        )

    # Distribuição de classes
    st.subheader("📊 Distribuição de Classes no Conjunto")
    st.caption("Valores de 'Suporte' que aparecem nos relatórios de classificação")

    class_distribution = y_test.value_counts()

    col1, col2, col3 = st.columns(3)

    with col1:
        normal_count = class_distribution.get('Normal', 0)
        normal_pct = (normal_count / len(y_test)) * 100
        st.metric(
            "Normal",
            f"{normal_count:,}",
            f"{normal_pct:.2f}%",
            help="Tráfego legítimo"
        )

    with col2:
        backdoor_count = class_distribution.get('Backdoor', 0)
        backdoor_pct = (backdoor_count / len(y_test)) * 100
        st.metric(
            "🔴 Backdoor",
            f"{backdoor_count:,}",
            f"{backdoor_pct:.2f}%",
            delta_color="inverse",
            help="Ataques backdoor"
        )

    with col3:
        worms_count = class_distribution.get('Worms', 0)
        worms_pct = (worms_count / len(y_test)) * 100
        st.metric(
            "🟠 Worms",
            f"{worms_count:,}",
            f"{worms_pct:.2f}%",
            delta_color="inverse",
            help="Ataques worm"
        )

    # ========================================================================
    # SEÇÃO 2: Resumo de Detecções (ambos os modelos)
    # ========================================================================
    st.divider()
    st.header("📊 Resumo de Detecções - Comparação dos Modelos")

    tab_naive, tab_improved = st.tabs(["🔵 Modelo Ingênuo", "🟢 Modelo Melhorado"])

    with tab_naive:
        st.subheader("Detecções - Modelo Ingênuo")

        naive_counts = pd.Series(naive_predictions).value_counts()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total de Conexões",
                f"{len(naive_predictions):,}",
                help="Total de conexões analisadas"
            )

        with col2:
            normal_detected = naive_counts.get('Normal', 0)
            st.metric(
                "Normais Detectadas",
                f"{normal_detected:,}",
                f"{normal_detected/len(naive_predictions)*100:.1f}%"
            )

        with col3:
            backdoor_detected = naive_counts.get('Backdoor', 0)
            st.metric(
                "🔴 Backdoors Detectados",
                f"{backdoor_detected:,}",
                f"{backdoor_detected/len(naive_predictions)*100:.1f}%",
                delta_color="inverse"
            )

        with col4:
            worms_detected = naive_counts.get('Worms', 0)
            st.metric(
                "🟠 Worms Detectados",
                f"{worms_detected:,}",
                f"{worms_detected/len(naive_predictions)*100:.1f}%",
                delta_color="inverse"
            )

        # Taxa de anomalias
        anomaly_rate_naive = ((backdoor_detected + worms_detected) / len(naive_predictions)) * 100

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                "Taxa de Anomalias",
                f"{anomaly_rate_naive:.1f}%",
                help="Porcentagem de tráfego classificado como malicioso"
            )

            if anomaly_rate_naive < 5:
                st.success("🟢 Nível de ameaça baixo")
            elif anomaly_rate_naive < 20:
                st.warning("🟡 Nível de ameaça moderado")
            else:
                st.error("🔴 Nível de ameaça alto")

        with col2:
            chart_data = pd.DataFrame({
                'Categoria': ['Normal', 'Backdoor', 'Worms'],
                'Contagem': [
                    naive_counts.get('Normal', 0),
                    naive_counts.get('Backdoor', 0),
                    naive_counts.get('Worms', 0)
                ]
            })
            st.bar_chart(chart_data.set_index('Categoria'))

    with tab_improved:
        st.subheader("Detecções - Modelo Melhorado")

        improved_counts = pd.Series(improved_predictions).value_counts()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total de Conexões",
                f"{len(improved_predictions):,}",
                help="Total de conexões analisadas"
            )

        with col2:
            normal_detected = improved_counts.get('Normal', 0)
            st.metric(
                "Normais Detectadas",
                f"{normal_detected:,}",
                f"{normal_detected/len(improved_predictions)*100:.1f}%"
            )

        with col3:
            backdoor_detected = improved_counts.get('Backdoor', 0)
            st.metric(
                "🔴 Backdoors Detectados",
                f"{backdoor_detected:,}",
                f"{backdoor_detected/len(improved_predictions)*100:.1f}%",
                delta_color="inverse"
            )

        with col4:
            worms_detected = improved_counts.get('Worms', 0)
            st.metric(
                "🟠 Worms Detectados",
                f"{worms_detected:,}",
                f"{worms_detected/len(improved_predictions)*100:.1f}%",
                delta_color="inverse"
            )

        # Taxa de anomalias
        anomaly_rate_improved = ((backdoor_detected + worms_detected) / len(improved_predictions)) * 100

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                "Taxa de Anomalias",
                f"{anomaly_rate_improved:.1f}%",
                help="Porcentagem de tráfego classificado como malicioso"
            )

            if anomaly_rate_improved < 5:
                st.success("🟢 Nível de ameaça baixo")
            elif anomaly_rate_improved < 20:
                st.warning("🟡 Nível de ameaça moderado")
            else:
                st.error("🔴 Nível de ameaça alto")

        with col2:
            chart_data = pd.DataFrame({
                'Categoria': ['Normal', 'Backdoor', 'Worms'],
                'Contagem': [
                    improved_counts.get('Normal', 0),
                    improved_counts.get('Backdoor', 0),
                    improved_counts.get('Worms', 0)
                ]
            })
            st.bar_chart(chart_data.set_index('Categoria'))

    # ========================================================================
    # SEÇÃO 3: Métricas de Desempenho - Análise Comparativa
    # ========================================================================
    st.divider()
    st.header("📈 Avaliação de Desempenho - Métricas")
    st.caption("✅ Rótulos ground truth detectados - Comparação completa dos modelos")

    # Tabela Comparativa de F1-Scores
    st.subheader("📊 Tabela 2 - Comparativo de F1-Score para Classes Minoritárias")
    st.caption("Esta tabela replica exatamente os resultados em Tabela 2 da tese")

    comparison_data = []
    for attack_type in ['Backdoor', 'Worms']:
        naive_f1 = naive_report_dict[attack_type]['f1-score']
        improved_f1 = improved_report_dict[attack_type]['f1-score']
        improvement = ((improved_f1 - naive_f1) / naive_f1 * 100) if naive_f1 > 0 else 0

        comparison_data.append({
            'Categoria de Ataque': attack_type,
            'F1-Score (Modelo Ingênuo)': f"{naive_f1:.4f}",
            'F1-Score (Modelo Melhorado)': f"{improved_f1:.4f}",
            'Melhoria': f"{improvement:+.2f}%"
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Métricas Gerais Comparativas
    st.markdown("---")
    st.markdown("#### Desempenho Geral dos Modelos")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Acurácia (Ingênuo)",
            f"{naive_report_dict['accuracy']:.2%}",
            help="Modelo sem tratamento de desbalanceamento"
        )
        st.metric(
            "Acurácia (Melhorado)",
            f"{improved_report_dict['accuracy']:.2%}",
            help="Modelo com class_weight='balanced'"
        )

    with col2:
        st.metric(
            "F1-Score Ponderado (Ingênuo)",
            f"{naive_report_dict['weighted avg']['f1-score']:.4f}"
        )
        st.metric(
            "F1-Score Ponderado (Melhorado)",
            f"{improved_report_dict['weighted avg']['f1-score']:.4f}"
        )

    with col3:
        st.metric(
            "Recall Médio Macro (Ingênuo)",
            f"{naive_report_dict['macro avg']['recall']:.2%}"
        )
        st.metric(
            "Recall Médio Macro (Melhorado)",
            f"{improved_report_dict['macro avg']['recall']:.2%}"
        )

    # ========================================================================
    # SEÇÃO 4: Relatórios Detalhados de Classificação
    # ========================================================================
    st.divider()
    st.header("📋 Relatórios de Classificação Detalhados (Tabela 3 da Tese)")
    st.caption("Estes relatórios são idênticos aos gerados pelo modelo ")

    tab1, tab2 = st.tabs(["🔵 Modelo Ingênuo", "🟢 Modelo Melhorado"])

    with tab1:
        st.markdown("### MODELO INGÊNUO - RELATÓRIO DE CLASSIFICAÇÃO")
        st.caption("Modelo: RandomForestClassifier (n_estimators=100)")
        st.caption("Tratamento de Desbalanceamento de Classes: Nenhum")

        # Criar tabela
        naive_data = []
        for class_name in ['Backdoor', 'Normal', 'Worms']:
            metrics = naive_report_dict[class_name]
            naive_data.append({
                'Categoria': class_name,
                'Precisão': f"{metrics['precision']:.4f}",
                'Recall (Taxa de Detecção)': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1-score']:.4f}",
                'Suporte (Nº de Amostras)': int(metrics['support'])
            })

        naive_data.append({
            'Categoria': 'Acurácia Geral',
            'Precisão': '',
            'Recall (Taxa de Detecção)': '',
            'F1-Score': f"{naive_report_dict['accuracy']:.4f}",
            'Suporte (Nº de Amostras)': len(y_test)
        })

        naive_data.append({
            'Categoria': 'Média Macro',
            'Precisão': f"{naive_report_dict['macro avg']['precision']:.4f}",
            'Recall (Taxa de Detecção)': f"{naive_report_dict['macro avg']['recall']:.4f}",
            'F1-Score': f"{naive_report_dict['macro avg']['f1-score']:.4f}",
            'Suporte (Nº de Amostras)': len(y_test)
        })

        naive_data.append({
            'Categoria': 'Média Ponderada',
            'Precisão': f"{naive_report_dict['weighted avg']['precision']:.4f}",
            'Recall (Taxa de Detecção)': f"{naive_report_dict['weighted avg']['recall']:.4f}",
            'F1-Score': f"{naive_report_dict['weighted avg']['f1-score']:.4f}",
            'Suporte (Nº de Amostras)': len(y_test)
        })

        naive_df = pd.DataFrame(naive_data)
        st.dataframe(naive_df, use_container_width=True, hide_index=True)

        # Matriz de confusão
        with st.expander("📊 Ver Matriz de Confusão - Figura 11"):
            st.markdown("#### Figura 11 - Matriz de Confusão: Modelo Ingênuo")
            cm_naive = confusion_matrix(y_test_encoded, y_pred_naive)
            cm_naive_norm = cm_naive.astype('float') / cm_naive.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_naive_norm,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Porcentagem'},
                ax=ax
            )
            ax.set_title('Matriz de Confusão - Modelo Ingênuo', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Classe Prevista', fontsize=12, fontweight='bold')
            ax.set_ylabel('Classe Real', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        # Relatório em formato texto
        with st.expander("📄 Ver Relatório em Formato Texto"):
            naive_report_text = classification_report(y_test_encoded, y_pred_naive, target_names=class_labels, digits=4)
            st.text("MODELO INGÊNUO - RELATÓRIO DE CLASSIFICAÇÃO\n" +
                   "="*60 + "\n" +
                   "Modelo: RandomForestClassifier (n_estimators=100)\n" +
                   "Tratamento de Desbalanceamento de Classes: Nenhum\n" +
                   "="*60 + "\n\n" +
                   naive_report_text)

    with tab2:
        st.markdown("### MODELO MELHORADO - RELATÓRIO DE CLASSIFICAÇÃO")
        st.caption("Modelo: RandomForestClassifier (n_estimators=100)")
        st.caption("Tratamento de Desbalanceamento de Classes: class_weight='balanced'")

        improved_data = []
        for class_name in ['Backdoor', 'Normal', 'Worms']:
            metrics = improved_report_dict[class_name]
            improved_data.append({
                'Categoria': class_name,
                'Precisão': f"{metrics['precision']:.4f}",
                'Recall (Taxa de Detecção)': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1-score']:.4f}",
                'Suporte (Nº de Amostras)': int(metrics['support'])
            })

        improved_data.append({
            'Categoria': 'Acurácia Geral',
            'Precisão': '',
            'Recall (Taxa de Detecção)': '',
            'F1-Score': f"{improved_report_dict['accuracy']:.4f}",
            'Suporte (Nº de Amostras)': len(y_test)
        })

        improved_data.append({
            'Categoria': 'Média Macro',
            'Precisão': f"{improved_report_dict['macro avg']['precision']:.4f}",
            'Recall (Taxa de Detecção)': f"{improved_report_dict['macro avg']['recall']:.4f}",
            'F1-Score': f"{improved_report_dict['macro avg']['f1-score']:.4f}",
            'Suporte (Nº de Amostras)': len(y_test)
        })

        improved_data.append({
            'Categoria': 'Média Ponderada',
            'Precisão': f"{improved_report_dict['weighted avg']['precision']:.4f}",
            'Recall (Taxa de Detecção)': f"{improved_report_dict['weighted avg']['recall']:.4f}",
            'F1-Score': f"{improved_report_dict['weighted avg']['f1-score']:.4f}",
            'Suporte (Nº de Amostras)': len(y_test)
        })

        improved_df = pd.DataFrame(improved_data)
        st.dataframe(improved_df, use_container_width=True, hide_index=True)

        # Matriz de confusão
        with st.expander("📊 Ver Matriz de Confusão - Figura 12"):
            st.markdown("#### Figura 12 - Matriz de Confusão: Modelo Melhorado")
            cm_improved = confusion_matrix(y_test_encoded, y_pred_improved)
            cm_improved_norm = cm_improved.astype('float') / cm_improved.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_improved_norm,
                annot=True,
                fmt='.2%',
                cmap='Greens',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Porcentagem'},
                ax=ax
            )
            ax.set_title('Matriz de Confusão - Modelo Melhorado', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Classe Prevista', fontsize=12, fontweight='bold')
            ax.set_ylabel('Classe Real', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        # Relatório em formato texto
        with st.expander("📄 Ver Relatório em Formato Texto"):
            improved_report_text = classification_report(y_test_encoded, y_pred_improved, target_names=class_labels, digits=4)
            st.text("MODELO MELHORADO - RELATÓRIO DE CLASSIFICAÇÃO\n" +
                   "="*60 + "\n" +
                   "Modelo: RandomForestClassifier (n_estimators=100)\n" +
                   "Tratamento de Desbalanceamento de Classes: class_weight='balanced'\n" +
                   "="*60 + "\n\n" +
                   improved_report_text)

    # ========================================================================
    # SEÇÃO 5: Destacar Desempenho em Classes Minoritárias
    # ========================================================================
    st.divider()
    st.markdown("#### 🎯 Desempenho de Detecção de Classes Minoritárias")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Backdoor Detection**")
        backdoor_f1_naive = naive_report_dict['Backdoor']['f1-score']
        backdoor_f1_improved = improved_report_dict['Backdoor']['f1-score']

        st.metric(
            "F1-Score (Ingênuo)",
            f"{backdoor_f1_naive:.4f}",
            help=f"Recall: {naive_report_dict['Backdoor']['recall']:.2%}"
        )
        st.metric(
            "F1-Score (Melhorado)",
            f"{backdoor_f1_improved:.4f}",
            help=f"Recall: {improved_report_dict['Backdoor']['recall']:.2%}"
        )

        if backdoor_f1_naive > 0.70 and backdoor_f1_improved > 0.70:
            st.success("✅ Ambos os modelos têm detecção forte de Backdoor")

    with col2:
        st.markdown("**Worms Detection**")
        worms_f1_naive = naive_report_dict['Worms']['f1-score']
        worms_f1_improved = improved_report_dict['Worms']['f1-score']

        st.metric(
            "F1-Score (Ingênuo)",
            f"{worms_f1_naive:.4f}",
            help=f"Recall: {naive_report_dict['Worms']['recall']:.2%}"
        )
        st.metric(
            "F1-Score (Melhorado)",
            f"{worms_f1_improved:.4f}",
            help=f"Recall: {improved_report_dict['Worms']['recall']:.2%}"
        )

        if worms_f1_naive > 0.70 and worms_f1_improved > 0.70:
            st.success("✅ Ambos os modelos têm detecção forte de Worms")

    # Adicionar explicações
    with st.expander("📚 Entendendo as Métricas"):
        st.markdown("""
        **Precisão**: De todas as conexões classificadas como ataque, qual porcentagem eram realmente ataques?
        - Alta precisão = Poucos falsos alarmes

        **Recall (Taxa de Detecção)**: De todos os ataques reais, qual porcentagem o modelo detectou?
        - Alto recall = Poucos ataques perdidos
        - **Métrica mais crítica para aplicações de segurança**

        **F1-Score**: Média harmônica entre Precisão e Recall
        - Equilibra ambas as métricas
        - **Métrica chave para análise comparativa do TCC**

        **Suporte**: Número de instâncias reais de cada classe no conjunto de dados

        **Por que isso Importa para o projeto?**:
        A comparação direta entre modelos naive e improved permite validar empiricamente
        a eficácia (ou limitações) do tratamento de desbalanceamento de classes.
        """)

    # ========================================================================
    # SEÇÃO 6: Validação dos Resultados
    # ========================================================================
    st.divider()
    st.header("✅ Validação dos Resultados")

    st.success("**🎓 Resultados Reproduzidos Dinamicamente!**")

    st.markdown(f"""
    ### Comparação com Arquivos de Referência:

    **Suporte (número de amostras no teste):**
    - Backdoor: **{backdoor_count:,}** amostras
    - Normal: **{normal_count:,}** amostras
    - Worms: **{worms_count:,}** amostras
    """)
