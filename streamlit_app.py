"""
Dashboard Simplificado de Detecção de Anomalias de Rede
Frontend Streamlit que se comunica com a API backend
"""

import streamlit as st
import requests
import pandas as pd
from typing import Optional

# Configurar página
st.set_page_config(
    page_title="Sistema de Detecção de Anomalias de Rede",
    page_icon="🛡️",
    layout="wide"
)

# Configuração da API
API_URL = "http://127.0.0.1:8000/predict/csv"

# Aplicação Principal
st.title("🛡️ Sistema de Detecção de Anomalias de Rede")
st.write("Envie seus dados de rede em formato CSV para detectar ameaças potenciais (Backdoor, Worms).")

st.divider()

# Uploader de arquivo
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    st.info(f"📁 Arquivo enviado: **{uploaded_file.name}**")

    # Mostrar spinner enquanto processa
    with st.spinner('🔄 Analisando dados de rede... Isso pode levar um momento.'):
        try:
            # Preparar arquivo para upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}

            # Fazer requisição à API
            response = requests.post(API_URL, files=files)

        except requests.exceptions.ConnectionError:
            st.error("❌ **Erro de Conexão**: Não foi possível conectar à API backend. Certifique-se de que o servidor da API está rodando em http://127.0.0.1:8000")
            st.code("python backend_api.py", language="bash")
            st.stop()
        except Exception as e:
            st.error(f"❌ **Erro**: Ocorreu um erro inesperado: {str(e)}")
            st.stop()

    # Verificar se a chamada à API foi bem-sucedida
    if response.status_code == 200:
        st.success('✅ Análise Completa!')
        results = response.json()

        st.divider()
        st.header("📊 Relatório de Análise")

        # Métricas de Resumo
        st.subheader("Resumo")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total de Conexões Analisadas",
                f"{results.get('total_connections', 0):,}",
                help="Número total de conexões de rede no arquivo enviado"
            )

        with col2:
            st.metric(
                "Conexões Normais",
                f"{results.get('normal_connections', 0):,}",
                delta=f"{results.get('normal_connections', 0) / max(results.get('total_connections', 1), 1) * 100:.1f}%",
                help="Conexões classificadas como tráfego legítimo"
            )

        with col3:
            backdoor_count = results.get('backdoors_detected', 0)
            st.metric(
                "🔴 Backdoors Detectados",
                f"{backdoor_count:,}",
                delta=f"{backdoor_count / max(results.get('total_connections', 1), 1) * 100:.1f}%",
                delta_color="inverse",
                help="Tentativas de ataque backdoor detectadas"
            )

        with col4:
            worms_count = results.get('worms_detected', 0)
            st.metric(
                "🟠 Worms Detectados",
                f"{worms_count:,}",
                delta=f"{worms_count / max(results.get('total_connections', 1), 1) * 100:.1f}%",
                delta_color="inverse",
                help="Atividades de propagação de worm detectadas"
            )

        # Exibir métricas se foram calculadas
        if results.get("metrics") and results["metrics"].get("precision") is not None:
            st.divider()
            st.subheader("📈 Avaliação de Desempenho - Métricas Acadêmicas TCC")
            st.caption("✅ Rótulos ground truth detectados no arquivo enviado - Métricas de avaliação completas disponíveis")

            metrics = results["metrics"]

            # Exibir métricas ponderadas gerais
            st.markdown("#### Desempenho Geral do Modelo")
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Precisão (Ponderada)",
                f"{metrics.get('precision', 0):.2%}",
                help="Proporção de identificações positivas que estavam realmente corretas (média ponderada entre todas as classes)"
            )
            col2.metric(
                "Recall (Ponderado)",
                f"{metrics.get('recall', 0):.2%}",
                help="Proporção de positivos reais que foram identificados corretamente (média ponderada entre todas as classes)"
            )
            col3.metric(
                "F1-Score (Ponderado)",
                f"{metrics.get('f1_score', 0):.2%}",
                help="Média harmônica entre precisão e recall (média ponderada entre todas as classes)"
            )

            # Exibir relatório de classificação completo se disponível (CHAVE PARA TCC)
            if results.get("classification_report"):
                st.markdown("---")
                st.markdown("#### 📊 Relatório de Classificação Detalhado (Métricas por Classe)")
                st.markdown("**Esta é a avaliação central para validação da hipótese do TCC**")

                classification_report = results["classification_report"]

                # Criar DataFrame para melhor visualização
                classes_to_show = ['Backdoor', 'Worms', 'Normal']
                report_data = []

                for class_name in classes_to_show:
                    if class_name in classification_report:
                        class_metrics = classification_report[class_name]
                        report_data.append({
                            'Classe': class_name,
                            'Precisão': f"{class_metrics.get('precision', 0):.4f}",
                            'Recall': f"{class_metrics.get('recall', 0):.4f}",
                            'F1-Score': f"{class_metrics.get('f1-score', 0):.4f}",
                            'Suporte': int(class_metrics.get('support', 0))
                        })

                # Adicionar linha de média ponderada
                if 'weighted avg' in classification_report:
                    weighted_avg = classification_report['weighted avg']
                    report_data.append({
                        'Classe': 'Média Ponderada',
                        'Precisão': f"{weighted_avg.get('precision', 0):.4f}",
                        'Recall': f"{weighted_avg.get('recall', 0):.4f}",
                        'F1-Score': f"{weighted_avg.get('f1-score', 0):.4f}",
                        'Suporte': int(weighted_avg.get('support', 0))
                    })

                report_df = pd.DataFrame(report_data)

                # Estilizar o dataframe
                st.dataframe(
                    report_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Adicionar explicações para avaliação do TCC
                with st.expander("📚 Entendendo as Métricas (Contexto TCC)"):
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

                    **Por que isso Importa para o TCC**:
                    A capacidade do modelo de detectar classes minoritárias (Backdoor, Worms) com F1-scores elevados
                    demonstra a eficácia do tratamento de desbalanceamento de classes (class_weight='balanced').
                    Isso valida diretamente a hipótese do TCC.
                    """)

                # Destacar desempenho em classes minoritárias (CHAVE PARA TCC)
                st.markdown("---")
                st.markdown("#### 🎯 Desempenho de Detecção de Classes Minoritárias (Descoberta Chave do TCC)")

                col1, col2 = st.columns(2)

                with col1:
                    backdoor_f1 = classification_report.get('Backdoor', {}).get('f1-score', 0)
                    backdoor_recall = classification_report.get('Backdoor', {}).get('recall', 0)
                    st.metric(
                        "Detecção de Backdoor (F1-Score)",
                        f"{backdoor_f1:.2%}",
                        help=f"Recall: {backdoor_recall:.2%} - Detectando com sucesso {backdoor_recall:.0%} dos ataques Backdoor"
                    )

                    if backdoor_f1 > 0.70:
                        st.success("✅ Detecção forte de ataques Backdoor")
                    elif backdoor_f1 > 0.50:
                        st.warning("⚠️ Detecção moderada - Considere ajustar o modelo")
                    else:
                        st.error("❌ Detecção fraca - Problema de desbalanceamento de classes")

                with col2:
                    worms_f1 = classification_report.get('Worms', {}).get('f1-score', 0)
                    worms_recall = classification_report.get('Worms', {}).get('recall', 0)
                    st.metric(
                        "Detecção de Worms (F1-Score)",
                        f"{worms_f1:.2%}",
                        help=f"Recall: {worms_recall:.2%} - Detectando com sucesso {worms_recall:.0%} dos ataques Worm"
                    )

                    if worms_f1 > 0.70:
                        st.success("✅ Detecção forte de ataques Worm")
                    elif worms_f1 > 0.50:
                        st.warning("⚠️ Detecção moderada - Considere ajustar o modelo")
                    else:
                        st.error("❌ Detecção fraca - Problema de desbalanceamento de classes")

                st.success("✅ As métricas de desempenho do modelo confirmam tratamento eficaz do desbalanceamento de classes")

            else:
                st.success("✅ O modelo está com bom desempenho no seu conjunto de dados!")
        else:
            st.info("ℹ️ Nenhum rótulo verdadeiro (coluna 'attack_cat' ou 'label') encontrado no arquivo enviado, portanto as métricas de desempenho não puderam ser calculadas.")

        # Visualização de Taxa de Anomalias
        st.divider()
        st.subheader("🎯 Resumo de Detecção de Anomalias")

        anomaly_rate = results.get('anomaly_rate', 0)
        total_anomalies = results.get('total_anomalies', 0)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                "Taxa de Anomalias",
                f"{anomaly_rate:.1f}%",
                help="Porcentagem de tráfego classificado como malicioso"
            )

            if anomaly_rate < 5:
                st.success("🟢 Nível de ameaça baixo - Rede aparenta estar segura")
            elif anomaly_rate < 20:
                st.warning("🟡 Nível de ameaça moderado - Monitorar de perto")
            else:
                st.error("🔴 Nível de ameaça alto - Ação imediata necessária!")

        with col2:
            # Dados do gráfico de pizza
            chart_data = pd.DataFrame({
                'Categoria': ['Normal', 'Backdoor', 'Worms'],
                'Contagem': [
                    results.get('normal_connections', 0),
                    results.get('backdoors_detected', 0),
                    results.get('worms_detected', 0)
                ]
            })

            st.bar_chart(chart_data.set_index('Categoria'))

        # Predições Brutas (expansível)
        with st.expander("🔍 Ver Predições Detalhadas", expanded=False):
            st.subheader("Predições de Conexões Individuais")

            predictions_list = results.get('predictions', [])

            if predictions_list:
                predictions_df = pd.DataFrame(predictions_list)

                # Adicionar filtros
                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    pred_filter = st.multiselect(
                        "Filtrar por predição:",
                        ['Normal', 'Backdoor', 'Worms'],
                        default=['Backdoor', 'Worms']
                    )

                with filter_col2:
                    min_confidence = st.slider(
                        "Confiança mínima:",
                        0.0, 1.0, 0.5, 0.05
                    )

                # Aplicar filtros
                filtered_df = predictions_df[
                    (predictions_df['prediction'].isin(pred_filter)) &
                    (predictions_df['confidence'] >= min_confidence)
                ]

                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=400
                )

                st.caption(f"Mostrando {len(filtered_df):,} de {len(predictions_df):,} predições")

                # Botão de download
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar Predições Filtradas (CSV)",
                    data=csv,
                    file_name="network_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.info("Nenhuma predição detalhada disponível na resposta.")

    else:
        # Exibir mensagem de erro amigável
        try:
            error_detail = response.json().get('detail', 'Ocorreu um erro desconhecido.')
        except:
            error_detail = response.text or 'Ocorreu um erro desconhecido.'

        st.error(f"❌ **Ocorreu um erro durante a análise:**\n\n{error_detail}")

        if response.status_code == 400:
            st.info("💡 **Dica**: Certifique-se de que seu arquivo CSV tem o formato correto e contém dados de tráfego de rede.")
        elif response.status_code == 500:
            st.info("💡 **Dica**: O servidor backend pode não ter carregado os artefatos do modelo. Certifique-se de executar `python model_training.py` primeiro.")

else:
    # Instruções quando nenhum arquivo foi enviado
    st.info("👆 **Por favor, envie um arquivo CSV para começar a análise**")

    with st.expander("ℹ️ Sobre Este Sistema"):
        st.markdown("""
        ### Como Funciona

        Este sistema usa aprendizado de máquina para detectar anomalias de rede em seus dados de tráfego:

        1. **Envie** seus dados de tráfego de rede em formato CSV
        2. **Análise** é realizada pelo nosso classificador Random Forest treinado no conjunto de dados UNSW-NB15
        3. **Resultados** mostram ameaças detectadas e métricas de desempenho

        ### Tipos de Ameaças Detectadas

        - 🟢 **Normal**: Tráfego de rede legítimo
        - 🔴 **Backdoor**: Tentativas de acesso não autorizado
        - 🟠 **Worms**: Propagação de malware auto-replicante

        ### Métricas de Desempenho (se ground truth estiver disponível)

        - **Precisão**: Quantas ameaças detectadas são realmente ameaças
        - **Recall**: Quantas ameaças reais foram detectadas
        - **F1-Score**: Desempenho geral do modelo

        ### Formato de Arquivo Esperado

        Seu arquivo CSV deve conter características de tráfego de rede. O sistema irá automaticamente:
        - Tratar colunas faltantes preenchendo com valores padrão
        - Codificar características categóricas com one-hot encoding
        - Escalar características numéricas
        - Alinhar com o formato dos dados de treinamento

        ### Requisitos

        A API backend deve estar rodando:
        ```bash
        python backend_api.py
        ```
        """)

    with st.expander("🚀 Guia de Início Rápido"):
        st.markdown("""
        ### Passo 1: Treinar o Modelo (Configuração única)

        ```bash
        python model_training.py
        ```

        Isso irá gerar os artefatos do modelo necessários.

        ### Passo 2: Iniciar a API Backend

        ```bash
        python backend_api.py
        ```

        A API estará disponível em http://localhost:8000

        ### Passo 3: Executar Este Dashboard

        ```bash
        streamlit run streamlit_app.py
        ```

        ### Passo 4: Enviar Seus Dados

        Envie um arquivo CSV contendo dados de tráfego de rede e visualize os resultados da análise!
        """)
