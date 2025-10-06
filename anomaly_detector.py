"""
Software de Análise de Segurança de Rede
Detecção de Malwares (Worms e Backdoor) com IA e Machine Learning
"""

from pathlib import Path
from io import BytesIO
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from collections import Counter

st.set_page_config(
    page_title="Análise de Segurança de Rede",
    page_icon="🛡️",
    layout="wide",
)

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load pre-trained model pipeline, label encoder, and expected columns."""
    model_files = sorted(BASE_DIR.glob("best_model_pipeline_*.joblib"))
    encoder_path = BASE_DIR / "label_encoder.joblib"
    columns_path = BASE_DIR / "model_columns.joblib"

    if not model_files or not encoder_path.exists() or not columns_path.exists():
        return None, None, None, None

    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    model_name = latest_model.stem.replace("best_model_pipeline_", "")

    pipeline = joblib.load(latest_model)
    encoder = joblib.load(encoder_path)
    columns = joblib.load(columns_path)

    return pipeline, encoder, columns, model_name


# --- Column Mapping ---
UNSW_CRITICAL_COLUMNS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'proto', 'service', 'state'
]


def suggest_column_mapping(uploaded_df: pd.DataFrame) -> dict:
    """Auto-suggest column mapping based on name similarity and data types."""
    mapping = {}
    uploaded_cols_lower = {col.lower(): col for col in uploaded_df.columns}

    for target_col in UNSW_CRITICAL_COLUMNS:
        # Exact match (case-insensitive)
        if target_col.lower() in uploaded_cols_lower:
            mapping[target_col] = uploaded_cols_lower[target_col.lower()]
        # Partial match
        else:
            for uploaded_col_lower, uploaded_col_original in uploaded_cols_lower.items():
                if target_col.replace('_', '') in uploaded_col_lower.replace('_', ''):
                    mapping[target_col] = uploaded_col_original
                    break

    return mapping


def preprocess_uploaded_data(df: pd.DataFrame, column_mapping: dict, expected_columns: pd.Index):
    """Preprocess uploaded data to match training format."""
    # Apply column mapping
    df_mapped = df.rename(columns={v: k for k, v in column_mapping.items()})

    # Add missing columns with defaults
    for col in UNSW_CRITICAL_COLUMNS:
        if col not in df_mapped.columns:
            if col in ['proto', 'service', 'state']:
                df_mapped[col] = 'unknown'
            else:
                df_mapped[col] = 0

    # Keep only mapped columns
    df_clean = df_mapped[UNSW_CRITICAL_COLUMNS].copy()

    # One-hot encode categorical features
    X = pd.get_dummies(df_clean)

    # Align with training columns
    X = X.reindex(columns=expected_columns, fill_value=0)

    return X


# --- Risk Scoring ---
def calculate_risk_score(prediction: str, confidence: float) -> tuple[float, str]:
    """
    Calculate risk score (0-100) and risk level for a connection.

    Risk formula:
    - Normal: low risk (inverse of confidence)
    - Backdoor: 40-85% risk (scales with confidence)
    - Worms: 50-95% risk (more dangerous, scales with confidence)
    """
    if prediction == 'Normal':
        # Lower confidence in "Normal" = higher risk (could be misclassified)
        risk_score = (1 - confidence) * 30  # Max 30% risk for uncertain normal traffic

        if risk_score < 10:
            risk_level = 'Baixo'
        elif risk_score < 20:
            risk_level = 'Médio'
        else:
            risk_level = 'Alto'

    elif prediction == 'Backdoor':
        # Backdoor: 40% base + up to 45% from confidence = 40-85%
        risk_score = 40 + (confidence * 45)

        if risk_score < 55:
            risk_level = 'Médio'
        elif risk_score < 75:
            risk_level = 'Alto'
        else:
            risk_level = 'Crítico'

    else:  # Worms
        # Worms: 50% base + up to 45% from confidence = 50-95%
        # Higher base because worms propagate
        risk_score = 50 + (confidence * 45)

        if risk_score < 65:
            risk_level = 'Médio'
        elif risk_score < 80:
            risk_level = 'Alto'
        else:
            risk_level = 'Crítico'

    return round(risk_score, 2), risk_level


def predict_future_risk(predictions: np.ndarray, probabilities: np.ndarray, label_encoder) -> float:
    """
    Predict future risk based on current anomaly trends.

    Future risk considers:
    - Current anomaly rate
    - Average confidence of anomalies
    - Presence of critical threats (Worms)
    """
    pred_labels = label_encoder.inverse_transform(predictions)

    total = len(pred_labels)
    if total == 0:
        return 0.0

    # Count anomalies
    backdoor_count = sum(pred_labels == 'Backdoor')
    worms_count = sum(pred_labels == 'Worms')
    anomaly_count = backdoor_count + worms_count

    # Current anomaly rate (0-100)
    anomaly_rate = (anomaly_count / total) * 100

    # Average confidence of anomaly predictions
    anomaly_indices = np.where((pred_labels == 'Backdoor') | (pred_labels == 'Worms'))[0]
    if len(anomaly_indices) > 0:
        anomaly_confidences = probabilities[anomaly_indices].max(axis=1)
        avg_anomaly_confidence = anomaly_confidences.mean()
    else:
        avg_anomaly_confidence = 0

    # Worm presence multiplier (worms spread = higher future risk)
    worm_multiplier = 1.5 if worms_count > 0 else 1.0

    # Future risk formula
    # Base: current anomaly rate
    # Boost: average confidence of threats (high confidence = real threats)
    # Worm factor: if worms present, increase future risk
    future_risk = (anomaly_rate * 0.6 + avg_anomaly_confidence * 40) * worm_multiplier

    # Cap at 100
    return min(round(future_risk, 2), 100.0)


# --- Report Statistics ---
def compute_report_statistics(df: pd.DataFrame, predictions: np.ndarray,
                              probabilities: np.ndarray, label_encoder) -> dict:
    """
    Compute all statistics required for the security report.

    Required metrics:
    - Total de conexões monitoradas
    - Conexões normais
    - Anomalias detectadas
    - Nível de confiança média da análise
    - Risco médio futuro esperado
    - Conexões suspeitas (alto risco)
    """
    pred_labels = label_encoder.inverse_transform(predictions)
    confidences = probabilities.max(axis=1)

    total = len(pred_labels)
    normal_count = sum(pred_labels == 'Normal')
    backdoor_count = sum(pred_labels == 'Backdoor')
    worms_count = sum(pred_labels == 'Worms')
    anomaly_count = backdoor_count + worms_count

    # Calculate average confidence
    avg_confidence = confidences.mean() * 100 if total > 0 else 0

    # Calculate future risk
    future_risk = predict_future_risk(predictions, probabilities, label_encoder)

    # Calculate risk scores for all connections
    risk_scores = []
    risk_levels = []
    for pred, conf in zip(pred_labels, confidences):
        score, level = calculate_risk_score(pred, conf)
        risk_scores.append(score)
        risk_levels.append(level)

    # Count high-risk connections (Alto or Crítico)
    high_risk_count = sum(1 for level in risk_levels if level in ['Alto', 'Crítico'])

    return {
        'total_connections': total,
        'normal_count': normal_count,
        'normal_pct': (normal_count / total * 100) if total > 0 else 0,
        'anomaly_count': anomaly_count,
        'anomaly_pct': (anomaly_count / total * 100) if total > 0 else 0,
        'backdoor_count': backdoor_count,
        'worms_count': worms_count,
        'avg_confidence': avg_confidence,
        'future_risk': future_risk,
        'high_risk_count': high_risk_count,
        'risk_scores': risk_scores,
        'risk_levels': risk_levels,
    }


# --- IP/Port Analysis ---
def analyze_network_topology(df_original: pd.DataFrame, predictions: np.ndarray,
                             probabilities: np.ndarray, label_encoder, column_mapping: dict) -> dict:
    """
    Analyze network topology to identify attack sources, targets, and patterns.

    Returns insights about:
    - Top attacking IPs (sources of anomalies)
    - Most targeted IPs (destinations of attacks)
    - Attack port analysis
    - Protocol distribution for attacks
    """
    pred_labels = label_encoder.inverse_transform(predictions)
    confidences = probabilities.max(axis=1)

    df_analysis = df_original.copy()
    df_analysis['prediction'] = pred_labels
    df_analysis['confidence'] = confidences

    # Filter to anomalies only
    anomaly_df = df_analysis[df_analysis['prediction'].isin(['Backdoor', 'Worms'])].copy()

    insights = {}

    # Try to find IP columns (common naming patterns)
    src_ip_candidates = ['srcip', 'src_ip', 'source_ip', 'saddr', 'src']
    dst_ip_candidates = ['dstip', 'dst_ip', 'dest_ip', 'daddr', 'dst', 'destination_ip']
    port_candidates = ['dport', 'dst_port', 'port', 'service_port', 'sport', 'src_port']

    src_ip_col = None
    dst_ip_col = None
    port_col = None

    # Find columns (case-insensitive)
    df_cols_lower = {col.lower(): col for col in df_original.columns}

    for candidate in src_ip_candidates:
        if candidate in df_cols_lower:
            src_ip_col = df_cols_lower[candidate]
            break

    for candidate in dst_ip_candidates:
        if candidate in df_cols_lower:
            dst_ip_col = df_cols_lower[candidate]
            break

    for candidate in port_candidates:
        if candidate in df_cols_lower:
            port_col = df_cols_lower[candidate]
            break

    # Analyze source IPs (attackers)
    if src_ip_col and src_ip_col in anomaly_df.columns and not anomaly_df.empty:
        top_attackers = anomaly_df.groupby(src_ip_col).agg({
            'prediction': 'count',
            'confidence': 'mean'
        }).rename(columns={'prediction': 'attack_count', 'confidence': 'avg_confidence'})
        top_attackers = top_attackers.sort_values('attack_count', ascending=False).head(10)
        top_attackers['avg_confidence'] = (top_attackers['avg_confidence'] * 100).round(1)
        insights['top_attackers'] = top_attackers

    # Analyze destination IPs (targets)
    if dst_ip_col and dst_ip_col in anomaly_df.columns and not anomaly_df.empty:
        top_targets = anomaly_df.groupby(dst_ip_col).agg({
            'prediction': 'count',
            'confidence': 'mean'
        }).rename(columns={'prediction': 'attack_count', 'confidence': 'avg_confidence'})
        top_targets = top_targets.sort_values('attack_count', ascending=False).head(10)
        top_targets['avg_confidence'] = (top_targets['avg_confidence'] * 100).round(1)
        insights['top_targets'] = top_targets

    # Analyze attack ports
    if port_col and port_col in anomaly_df.columns and not anomaly_df.empty:
        port_analysis = anomaly_df.groupby([port_col, 'prediction']).size().reset_index(name='count')
        top_ports = port_analysis.groupby(port_col)['count'].sum().sort_values(ascending=False).head(15)
        insights['top_ports'] = top_ports
        insights['port_by_attack'] = port_analysis

    # Protocol analysis for anomalies
    proto_col = column_mapping.get('proto')
    if proto_col and proto_col in anomaly_df.columns and not anomaly_df.empty:
        proto_dist = anomaly_df.groupby(['prediction', proto_col]).size().reset_index(name='count')
        insights['protocol_distribution'] = proto_dist

    # Attack patterns (repeated connections)
    if src_ip_col and dst_ip_col and src_ip_col in anomaly_df.columns and dst_ip_col in anomaly_df.columns:
        connection_pairs = anomaly_df.groupby([src_ip_col, dst_ip_col, 'prediction']).size()
        repeated_attacks = connection_pairs[connection_pairs > 1].sort_values(ascending=False).head(10)
        if len(repeated_attacks) > 0:
            insights['repeated_attacks'] = repeated_attacks

    return insights


# --- Time-Series Analysis ---
def analyze_time_trends(df_original: pd.DataFrame, predictions: np.ndarray,
                       probabilities: np.ndarray, label_encoder) -> dict:
    """
    Analyze temporal patterns in network traffic and anomalies.

    Detects:
    - Anomaly rate over time
    - Attack spikes
    - Trend direction (increasing/decreasing)
    - Peak attack periods
    """
    pred_labels = label_encoder.inverse_transform(predictions)

    df_analysis = df_original.copy()
    df_analysis['prediction'] = pred_labels
    df_analysis['is_anomaly'] = df_analysis['prediction'].isin(['Backdoor', 'Worms'])

    insights = {}

    # Try to find timestamp column
    timestamp_candidates = ['timestamp', 'time', 'datetime', 'date', 'ts', 'stime', 'ltime']
    timestamp_col = None

    df_cols_lower = {col.lower(): col for col in df_original.columns}

    for candidate in timestamp_candidates:
        if candidate in df_cols_lower:
            timestamp_col = df_cols_lower[candidate]
            break

    if timestamp_col and timestamp_col in df_analysis.columns:
        # Try to parse timestamp
        try:
            df_analysis['parsed_time'] = pd.to_datetime(df_analysis[timestamp_col], errors='coerce')

            # Remove rows with invalid timestamps
            df_time = df_analysis[df_analysis['parsed_time'].notna()].copy()

            if len(df_time) > 0:
                # Sort by time
                df_time = df_time.sort_values('parsed_time')

                # Time range
                time_range = df_time['parsed_time'].max() - df_time['parsed_time'].min()
                insights['time_range'] = time_range
                insights['start_time'] = df_time['parsed_time'].min()
                insights['end_time'] = df_time['parsed_time'].max()

                # Determine appropriate time bucket
                if time_range > timedelta(days=7):
                    freq = 'D'  # Daily
                    freq_label = 'dia'
                elif time_range > timedelta(hours=24):
                    freq = 'H'  # Hourly
                    freq_label = 'hora'
                elif time_range > timedelta(hours=1):
                    freq = '10T'  # 10 minutes
                    freq_label = '10 minutos'
                else:
                    freq = 'T'  # Minute
                    freq_label = 'minuto'

                insights['freq_label'] = freq_label

                # Resample data by time buckets
                df_time.set_index('parsed_time', inplace=True)

                # Count total connections and anomalies per bucket
                time_series = df_time.resample(freq).agg({
                    'prediction': 'count',
                    'is_anomaly': 'sum'
                }).rename(columns={'prediction': 'total', 'is_anomaly': 'anomalies'})

                time_series['anomaly_rate'] = (time_series['anomalies'] / time_series['total'] * 100).fillna(0)

                insights['time_series'] = time_series.reset_index()

                # Detect spikes (anomaly rate > 2x average)
                avg_rate = time_series['anomaly_rate'].mean()
                spikes = time_series[time_series['anomaly_rate'] > avg_rate * 2]
                if len(spikes) > 0:
                    insights['spikes'] = spikes.reset_index()

                # Trend analysis (simple linear trend)
                if len(time_series) > 2:
                    x = np.arange(len(time_series))
                    y = time_series['anomaly_rate'].values

                    # Simple linear regression
                    slope = np.polyfit(x, y, 1)[0]

                    if slope > 0.5:
                        trend = "crescente"
                        trend_emoji = "📈"
                    elif slope < -0.5:
                        trend = "decrescente"
                        trend_emoji = "📉"
                    else:
                        trend = "estável"
                        trend_emoji = "➡️"

                    insights['trend'] = trend
                    insights['trend_emoji'] = trend_emoji
                    insights['trend_slope'] = slope

                # Attack type over time
                attack_time_series = df_time[df_time['prediction'].isin(['Backdoor', 'Worms'])].resample(freq)['prediction'].value_counts().unstack(fill_value=0)
                if not attack_time_series.empty:
                    insights['attack_time_series'] = attack_time_series.reset_index()

        except Exception as e:
            insights['error'] = f"Erro ao processar timestamp: {str(e)}"

    return insights


# --- Model Explainability ---
def get_feature_importance_for_sample(model_pipeline, sample_features: pd.DataFrame,
                                     expected_columns: pd.Index, top_n: int = 5) -> pd.DataFrame:
    """
    Get feature importance for a specific sample using feature weights.

    For tree-based models, uses feature importances weighted by feature values.
    Returns top N most influential features for this prediction.
    """
    classifier = model_pipeline.named_steps.get('classifier')

    if not hasattr(classifier, 'feature_importances_'):
        return pd.DataFrame()

    # Get feature importances
    importances = classifier.feature_importances_

    # Get feature values (after scaling)
    scaler = model_pipeline.named_steps.get('scaler')
    if scaler:
        scaled_values = scaler.transform(sample_features)
    else:
        scaled_values = sample_features.values

    # Calculate influence (importance * abs(value))
    influence = importances * np.abs(scaled_values[0])

    # Create dataframe
    feature_influence = pd.DataFrame({
        'Feature': expected_columns,
        'Importance': importances,
        'Value': scaled_values[0],
        'Influence': influence
    }).sort_values('Influence', ascending=False).head(top_n)

    return feature_influence


# --- UI: Main Report ---
def render_security_report(df_original: pd.DataFrame, predictions: np.ndarray,
                          probabilities: np.ndarray, label_encoder):
    """Render the main security analysis report."""
    st.header("📊 Relatório de Análise de Segurança")

    # Compute statistics
    stats = compute_report_statistics(df_original, predictions, probabilities, label_encoder)

    # Generate timestamp
    analysis_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    st.caption(f"Análise realizada em: {analysis_time}")

    st.divider()

    # Key metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🔍 Total de Conexões Monitoradas",
            f"{stats['total_connections']:,}",
            help="Número total de registros analisados no dataset"
        )
        st.metric(
            "✅ Conexões Normais",
            f"{stats['normal_count']:,}",
            delta=f"{stats['normal_pct']:.1f}%",
            help="Conexões classificadas como tráfego legítimo"
        )

    with col2:
        st.metric(
            "⚠️ Anomalias Detectadas",
            f"{stats['anomaly_count']:,}",
            delta=f"{stats['anomaly_pct']:.1f}%",
            delta_color="inverse",
            help="Total de ameaças detectadas (Backdoor + Worms)"
        )
        st.metric(
            "🎯 Nível de Confiança Média",
            f"{stats['avg_confidence']:.1f}%",
            help="Confiança média do modelo nas predições realizadas"
        )

    with col3:
        # Future risk with color coding
        risk_color = "🟢" if stats['future_risk'] < 30 else "🟡" if stats['future_risk'] < 60 else "🔴"
        st.metric(
            "🔮 Risco Médio Futuro Esperado",
            f"{risk_color} {stats['future_risk']:.1f}%",
            help="Estimativa de risco futuro baseada nas anomalias detectadas e propagação de worms"
        )
        st.metric(
            "🚨 Conexões Suspeitas (Alto Risco)",
            f"{stats['high_risk_count']:,}",
            help="Conexões classificadas com nível de risco Alto ou Crítico"
        )

    st.divider()

    # Detailed breakdown
    st.subheader("Detalhamento das Anomalias")

    col_a, col_b = st.columns(2)

    with col_a:
        # Pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = [stats['normal_count'], stats['backdoor_count'], stats['worms_count']]
        labels = ['Normal', 'Backdoor', 'Worms']
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        explode = (0, 0.1, 0.1)  # Explode anomalies

        ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
               explode=explode, startangle=90, shadow=True)
        ax.set_title('Distribuição de Tráfego')
        st.pyplot(fig, use_container_width=True)

    with col_b:
        # Bar chart of anomaly types
        fig, ax = plt.subplots(figsize=(6, 4))
        anomaly_data = pd.DataFrame({
            'Tipo': ['Backdoor', 'Worms'],
            'Quantidade': [stats['backdoor_count'], stats['worms_count']]
        })
        sns.barplot(data=anomaly_data, x='Tipo', y='Quantidade', palette=['#e74c3c', '#f39c12'], ax=ax)
        ax.set_title('Detecção de Malwares')
        ax.set_ylabel('Número de Detecções')
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig, use_container_width=True)

    return stats


def render_suspicious_connections(df_original: pd.DataFrame, predictions: np.ndarray,
                                  probabilities: np.ndarray, label_encoder, risk_scores, risk_levels):
    """Display table of high-risk connections."""
    st.header("🚨 Conexões Suspeitas (Alto Risco)")

    pred_labels = label_encoder.inverse_transform(predictions)
    confidences = probabilities.max(axis=1)

    # Build results dataframe
    results_df = df_original.copy()
    results_df['Classificação'] = pred_labels
    results_df['Confiança'] = (confidences * 100).round(1)
    results_df['Risco (%)'] = risk_scores
    results_df['Nível de Risco'] = risk_levels

    # Filter high-risk only
    high_risk_df = results_df[results_df['Nível de Risco'].isin(['Alto', 'Crítico'])].copy()
    high_risk_df = high_risk_df.sort_values('Risco (%)', ascending=False)

    if high_risk_df.empty:
        st.success("✅ Nenhuma conexão de alto risco detectada!")
        return

    st.warning(f"⚠️ {len(high_risk_df)} conexões suspeitas identificadas")

    # Priority columns to show
    priority_cols = ['Classificação', 'Confiança', 'Risco (%)', 'Nível de Risco']

    # Add network-specific columns if available
    network_cols = []
    for col in ['proto', 'service', 'state', 'sbytes', 'dbytes', 'dur']:
        if col in high_risk_df.columns:
            network_cols.append(col)

    display_cols = priority_cols + network_cols
    available_display_cols = [c for c in display_cols if c in high_risk_df.columns]

    # Display table
    st.dataframe(
        high_risk_df[available_display_cols].head(100),
        use_container_width=True,
        height=400
    )

    # Download button
    csv = high_risk_df.to_csv(index=False)
    st.download_button(
        label="📥 Baixar Conexões Suspeitas (CSV)",
        data=csv,
        file_name=f"conexoes_suspeitas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_detailed_results(df_original: pd.DataFrame, predictions: np.ndarray,
                           probabilities: np.ndarray, label_encoder, risk_scores, risk_levels):
    """Display detailed detection results with filters."""
    st.header("🔍 Resultados Detalhados da Análise")

    pred_labels = label_encoder.inverse_transform(predictions)
    confidences = probabilities.max(axis=1)

    results_df = df_original.copy()
    results_df['Classificação'] = pred_labels
    results_df['Confiança (%)'] = (confidences * 100).round(1)
    results_df['Risco (%)'] = risk_scores
    results_df['Nível de Risco'] = risk_levels

    # Filters
    st.subheader("Filtros")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        class_filter = st.multiselect(
            "Classificação",
            ['Normal', 'Backdoor', 'Worms'],
            default=['Backdoor', 'Worms']
        )

    with filter_col2:
        risk_filter = st.multiselect(
            "Nível de Risco",
            ['Baixo', 'Médio', 'Alto', 'Crítico'],
            default=['Alto', 'Crítico']
        )

    with filter_col3:
        min_confidence = st.slider("Confiança Mínima (%)", 0, 100, 50, 5)

    # Apply filters
    filtered_df = results_df[
        (results_df['Classificação'].isin(class_filter)) &
        (results_df['Nível de Risco'].isin(risk_filter)) &
        (results_df['Confiança (%)'] >= min_confidence)
    ].sort_values('Risco (%)', ascending=False)

    st.info(f"Exibindo {len(filtered_df):,} de {len(results_df):,} conexões")

    st.dataframe(filtered_df, use_container_width=True, height=400)

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Baixar Resultados Filtrados (CSV)",
        data=csv,
        file_name=f"analise_detalhada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_network_topology_analysis(df_original: pd.DataFrame, predictions: np.ndarray,
                                     probabilities: np.ndarray, label_encoder, column_mapping: dict):
    """Render IP/Port analysis and attack topology."""
    st.header("🌐 Análise de Topologia de Rede")

    insights = analyze_network_topology(df_original, predictions, probabilities, label_encoder, column_mapping)

    if not insights:
        st.info("💡 Adicione colunas de IP (srcip/dstip) ou portas (dport/sport) para análise de topologia detalhada.")
        return

    # Top Attackers
    if 'top_attackers' in insights and not insights['top_attackers'].empty:
        st.subheader("🔴 Top IPs Atacantes (Fontes de Anomalias)")
        st.caption("Endereços IP que originaram mais ataques detectados")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_att = insights['top_attackers'].head(10)
            ax.barh(range(len(top_att)), top_att['attack_count'].values, color='#e74c3c')
            ax.set_yticks(range(len(top_att)))
            ax.set_yticklabels(top_att.index)
            ax.set_xlabel('Número de Ataques')
            ax.set_title('Top 10 IPs Atacantes')
            ax.invert_yaxis()

            for i, v in enumerate(top_att['attack_count'].values):
                ax.text(v + 0.5, i, str(int(v)), va='center')

            st.pyplot(fig, use_container_width=True)

        with col2:
            st.dataframe(insights['top_attackers'], use_container_width=True)

            st.metric(
                "Ação Recomendada",
                f"Bloquear {min(3, len(insights['top_attackers']))} IPs",
                help="Bloqueie os IPs com mais ataques no firewall"
            )

    # Top Targets
    if 'top_targets' in insights and not insights['top_targets'].empty:
        st.subheader("🎯 Top IPs Alvos (Destinos de Ataques)")
        st.caption("Endereços IP mais visados por atacantes")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_tgt = insights['top_targets'].head(10)
            ax.barh(range(len(top_tgt)), top_tgt['attack_count'].values, color='#f39c12')
            ax.set_yticks(range(len(top_tgt)))
            ax.set_yticklabels(top_tgt.index)
            ax.set_xlabel('Número de Ataques Recebidos')
            ax.set_title('Top 10 IPs Alvos')
            ax.invert_yaxis()

            for i, v in enumerate(top_tgt['attack_count'].values):
                ax.text(v + 0.5, i, str(int(v)), va='center')

            st.pyplot(fig, use_container_width=True)

        with col2:
            st.dataframe(insights['top_targets'], use_container_width=True)

            st.metric(
                "Ação Recomendada",
                "Reforçar proteção",
                help="Implementar proteção adicional nestes hosts"
            )

    # Port Analysis
    if 'top_ports' in insights and not insights['top_ports'].empty:
        st.subheader("🔌 Análise de Portas de Ataque")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            top_ports_data = insights['top_ports'].head(15)
            ax.bar(range(len(top_ports_data)), top_ports_data.values, color='#3498db')
            ax.set_xticks(range(len(top_ports_data)))
            ax.set_xticklabels(top_ports_data.index, rotation=45, ha='right')
            ax.set_ylabel('Número de Ataques')
            ax.set_title('Top 15 Portas Atacadas')

            for i, v in enumerate(top_ports_data.values):
                ax.text(i, v + 1, str(int(v)), ha='center', va='bottom', fontsize=8)

            st.pyplot(fig, use_container_width=True)

        with col2:
            # Port by attack type
            if 'port_by_attack' in insights:
                st.write("**Portas por Tipo de Ataque:**")
                port_pivot = insights['port_by_attack'].pivot_table(
                    index=insights['port_by_attack'].columns[0],
                    columns='prediction',
                    values='count',
                    fill_value=0
                ).head(15)
                st.dataframe(port_pivot, use_container_width=True)

    # Repeated Attacks
    if 'repeated_attacks' in insights and not insights['repeated_attacks'].empty:
        st.subheader("🔁 Padrões de Ataque Repetidos")
        st.caption("Conexões repetidas entre mesmos IPs (possível varredura ou ataque coordenado)")

        repeated_df = insights['repeated_attacks'].reset_index()
        repeated_df.columns = ['IP Origem', 'IP Destino', 'Tipo', 'Tentativas']
        st.dataframe(repeated_df.head(10), use_container_width=True)

        st.warning(f"⚠️ {len(repeated_df)} padrões de ataque repetido detectados - possível varredura ou ataque coordenado")


def render_time_series_analysis(df_original: pd.DataFrame, predictions: np.ndarray,
                                probabilities: np.ndarray, label_encoder):
    """Render time-series trend analysis."""
    st.header("📈 Análise Temporal de Ataques")

    insights = analyze_time_trends(df_original, predictions, probabilities, label_encoder)

    if 'error' in insights:
        st.warning(f"⚠️ {insights['error']}")
        return

    if 'time_series' not in insights:
        st.info("💡 Adicione uma coluna de timestamp (timestamp, time, datetime) para análise temporal.")
        return

    # Time range and trend
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Período Analisado",
            f"{insights['time_range'].days} dias" if insights['time_range'].days > 0
            else f"{insights['time_range'].seconds // 3600} horas",
            help=f"De {insights['start_time']} até {insights['end_time']}"
        )

    with col2:
        if 'trend' in insights:
            st.metric(
                "Tendência de Ataques",
                f"{insights['trend_emoji']} {insights['trend'].capitalize()}",
                help=f"Slope: {insights['trend_slope']:.3f}"
            )

    with col3:
        if 'spikes' in insights:
            st.metric(
                "Picos Detectados",
                len(insights['spikes']),
                help="Momentos com taxa de anomalias acima de 2x a média"
            )

    # Time series plot
    st.subheader("Taxa de Anomalias ao Longo do Tempo")

    time_series = insights['time_series']

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot anomaly rate
    ax.plot(time_series['parsed_time'], time_series['anomaly_rate'],
            color='#e74c3c', linewidth=2, label='Taxa de Anomalias (%)')
    ax.fill_between(time_series['parsed_time'], 0, time_series['anomaly_rate'],
                     color='#e74c3c', alpha=0.3)

    # Mark spikes
    if 'spikes' in insights and not insights['spikes'].empty:
        spike_times = insights['spikes']['parsed_time']
        spike_rates = insights['spikes']['anomaly_rate']
        ax.scatter(spike_times, spike_rates, color='red', s=100, zorder=5,
                  label='Picos de Ataque', marker='^')

    ax.set_xlabel(f'Tempo (por {insights["freq_label"]})')
    ax.set_ylabel('Taxa de Anomalias (%)')
    ax.set_title('Evolução Temporal das Anomalias')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    # Attack type over time
    if 'attack_time_series' in insights and not insights['attack_time_series'].empty:
        st.subheader("Evolução por Tipo de Ataque")

        attack_ts = insights['attack_time_series']

        fig, ax = plt.subplots(figsize=(12, 5))

        if 'Backdoor' in attack_ts.columns:
            ax.plot(attack_ts['parsed_time'], attack_ts['Backdoor'],
                   color='#e74c3c', linewidth=2, marker='o', label='Backdoor')

        if 'Worms' in attack_ts.columns:
            ax.plot(attack_ts['parsed_time'], attack_ts['Worms'],
                   color='#f39c12', linewidth=2, marker='s', label='Worms')

        ax.set_xlabel(f'Tempo (por {insights["freq_label"]})')
        ax.set_ylabel('Número de Ataques')
        ax.set_title('Detecção de Malwares ao Longo do Tempo')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

    # Spike details
    if 'spikes' in insights and not insights['spikes'].empty:
        st.subheader("🚨 Detalhes dos Picos de Ataque")

        spikes = insights['spikes'][['parsed_time', 'total', 'anomalies', 'anomaly_rate']].copy()
        spikes.columns = ['Momento', 'Total Conexões', 'Anomalias', 'Taxa (%)']
        spikes['Taxa (%)'] = spikes['Taxa (%)'].round(1)

        st.dataframe(spikes, use_container_width=True)

        st.error(f"⚠️ {len(spikes)} picos de ataque detectados - requer investigação urgente!")


def render_explainability_view(df_original: pd.DataFrame, predictions: np.ndarray,
                               probabilities: np.ndarray, label_encoder, model, expected_columns,
                               column_mapping: dict):
    """Render model explainability for individual predictions."""
    st.header("🔍 Explicabilidade do Modelo")
    st.caption("Entenda por que cada conexão foi classificada de determinada forma")

    pred_labels = label_encoder.inverse_transform(predictions)
    confidences = probabilities.max(axis=1)

    # Filter to anomalies
    anomaly_indices = np.where(pred_labels != 'Normal')[0]

    if len(anomaly_indices) == 0:
        st.success("✅ Nenhuma anomalia detectada para explicar.")
        return

    # Select a sample
    st.subheader("Selecione uma Conexão para Análise")

    format_func = lambda idx: f"Conexão #{idx} - {pred_labels[idx]} (Confiança: {confidences[idx]*100:.1f}%)"

    selected_idx = st.selectbox(
        "Escolha uma anomalia",
        anomaly_indices[:100],  # Limit to first 100 for performance
        format_func=format_func
    )

    # Get prediction details
    pred_label = pred_labels[selected_idx]
    confidence = confidences[selected_idx]
    probs = probabilities[selected_idx]

    st.divider()

    # Prediction summary
    col1, col2, col3 = st.columns(3)

    col1.metric("Classificação", pred_label)
    col2.metric("Confiança", f"{confidence*100:.1f}%")

    # Risk
    from anomaly_detector import calculate_risk_score
    risk_score, risk_level = calculate_risk_score(pred_label, confidence)
    col3.metric("Nível de Risco", f"{risk_level} ({risk_score:.1f}%)")

    # Probability distribution
    st.subheader("Distribuição de Probabilidades")

    prob_df = pd.DataFrame({
        'Classe': label_encoder.classes_,
        'Probabilidade (%)': (probs * 100).round(2)
    }).sort_values('Probabilidade (%)', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#e74c3c' if c == pred_label else '#95a5a6' for c in prob_df['Classe']]
    ax.bar(prob_df['Classe'], prob_df['Probabilidade (%)'], color=colors)
    ax.set_ylabel('Probabilidade (%)')
    ax.set_title('Distribuição de Probabilidades para Esta Conexão')

    for i, (cls, prob) in enumerate(zip(prob_df['Classe'], prob_df['Probabilidade (%)'])):
        ax.text(i, prob + 1, f'{prob:.1f}%', ha='center', va='bottom')

    st.pyplot(fig, use_container_width=True)

    # Feature influence
    st.subheader("Principais Features que Influenciaram a Decisão")
    st.caption("Features com maior influência na classificação desta conexão")

    # Preprocess the single sample
    sample_df = df_original.iloc[[selected_idx]]
    from anomaly_detector import preprocess_uploaded_data
    X_sample = preprocess_uploaded_data(sample_df, column_mapping, expected_columns)

    # Get feature influence
    feature_influence = get_feature_importance_for_sample(model, X_sample, expected_columns, top_n=10)

    if not feature_influence.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_influence)), feature_influence['Influence'].values, color='#3498db')
        ax.set_yticks(range(len(feature_influence)))
        ax.set_yticklabels(feature_influence['Feature'].values)
        ax.set_xlabel('Influência na Decisão')
        ax.set_title(f'Top 10 Features Mais Influentes para Classificação "{pred_label}"')
        ax.invert_yaxis()

        st.pyplot(fig, use_container_width=True)

        # Show feature values
        with st.expander("Ver Valores das Features"):
            st.dataframe(feature_influence, use_container_width=True)

    # Original connection data
    with st.expander("Ver Dados Brutos da Conexão"):
        st.json(sample_df.iloc[0].to_dict())


# --- Upload Section ---
def render_upload_section():
    """Render file upload interface."""
    st.title("🛡️ Software de Análise de Segurança de Rede")
    st.caption("Detecção de Malwares (Worms e Backdoor) com IA e Machine Learning")

    uploaded_file = st.file_uploader(
        "📁 Fazer Upload de Dados de Rede (CSV ou Parquet)",
        type=['csv', 'parquet'],
        help="Envie seus logs de tráfego de rede para análise de segurança"
    )

    return uploaded_file


def render_column_mapping(df: pd.DataFrame):
    """Interactive column mapping interface."""
    st.subheader("📋 Mapeamento de Colunas")
    st.caption("Configure o mapeamento das colunas do seu dataset")

    suggested_mapping = suggest_column_mapping(df)
    uploaded_cols = ['(nenhum)'] + list(df.columns)
    mapping = {}

    with st.expander("Configurar Mapeamento de Colunas", expanded=True):
        st.info("💡 O sistema detectou automaticamente algumas colunas. Ajuste conforme necessário.")

        col1, col2 = st.columns(2)

        for idx, target_col in enumerate(UNSW_CRITICAL_COLUMNS):
            container = col1 if idx % 2 == 0 else col2

            default_idx = 0
            if target_col in suggested_mapping:
                try:
                    default_idx = uploaded_cols.index(suggested_mapping[target_col])
                except ValueError:
                    default_idx = 0

            selected = container.selectbox(
                f"{target_col}",
                uploaded_cols,
                index=default_idx,
                key=f"map_{target_col}"
            )

            if selected != '(nenhum)':
                mapping[target_col] = selected

    return mapping


# --- Main App ---
def main():
    # Load pre-trained model
    model, encoder, expected_columns, model_name = load_model_artifacts()

    if model is None:
        st.error("""
        ⚠️ **Modelo pré-treinado não encontrado!**

        Execute o script de treinamento para gerar os artefatos necessários:
        ```bash
        python model_training.py
        ```

        Arquivos necessários:
        - `best_model_pipeline_*.joblib`
        - `label_encoder.joblib`
        - `model_columns.joblib`
        """)
        return

    st.sidebar.success(f"✅ Modelo: **{model_name}**")
    st.sidebar.caption("Treinado com dataset UNSW-NB15")

    # Upload section
    uploaded_file = render_upload_section()

    if uploaded_file is None:
        st.info("👆 Faça upload de um dataset para iniciar a análise de segurança")

        with st.expander("ℹ️ Sobre Esta Ferramenta"):
            st.markdown("""
            ### Objetivo
            Software de análise de segurança de rede focado na detecção de malwares do tipo **Worm** e **Backdoor**
            utilizando Inteligência Artificial e técnicas de Machine Learning.

            ### Classes Detectadas
            - 🟢 **Normal**: Tráfego legítimo de rede
            - 🔴 **Backdoor**: Tentativas de ataque backdoor
            - 🟠 **Worms**: Atividade de propagação de worms

            ### Relatório Gerado
            Após a análise, o sistema fornece um relatório gráfico completo com:
            - Total de conexões monitoradas
            - Conexões normais
            - Anomalias detectadas
            - Nível de confiança média da análise
            - Risco médio futuro esperado
            - Conexões suspeitas (alto risco)

            ### Formatos Suportados
            - Arquivos CSV
            - Arquivos Parquet
            """)
        return

    # Load uploaded data
    try:
        if uploaded_file.name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_parquet(uploaded_file)

        st.success(f"✅ {len(df_uploaded):,} registros carregados de {uploaded_file.name}")
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return

    # Column mapping
    column_mapping = render_column_mapping(df_uploaded)

    if st.button("🚀 Iniciar Análise de Segurança", type="primary", use_container_width=True):
        with st.spinner("Processando dados e detectando anomalias..."):
            try:
                # Preprocess
                X_processed = preprocess_uploaded_data(df_uploaded, column_mapping, expected_columns)

                # Predict
                predictions = model.predict(X_processed)
                probabilities = model.predict_proba(X_processed)

                # Store in session state
                st.session_state['predictions'] = predictions
                st.session_state['probabilities'] = probabilities
                st.session_state['df_uploaded'] = df_uploaded
                st.session_state['column_mapping'] = column_mapping

                st.success("✅ Análise concluída com sucesso!")

            except Exception as e:
                st.error(f"Erro durante a análise: {e}")
                st.exception(e)
                return

    # Display results if available
    if 'predictions' in st.session_state:
        st.divider()

        # Render main security report
        stats = render_security_report(
            st.session_state['df_uploaded'],
            st.session_state['predictions'],
            st.session_state['probabilities'],
            encoder
        )

        st.divider()

        # Tabs for additional views
        tabs = st.tabs([
            "Conexões Suspeitas",
            "Análise de Rede (IPs/Portas)",
            "Análise Temporal",
            "Explicabilidade",
            "Resultados Detalhados"
        ])

        with tabs[0]:
            render_suspicious_connections(
                st.session_state['df_uploaded'],
                st.session_state['predictions'],
                st.session_state['probabilities'],
                encoder,
                stats['risk_scores'],
                stats['risk_levels']
            )

        with tabs[1]:
            render_network_topology_analysis(
                st.session_state['df_uploaded'],
                st.session_state['predictions'],
                st.session_state['probabilities'],
                encoder,
                st.session_state['column_mapping']
            )

        with tabs[2]:
            render_time_series_analysis(
                st.session_state['df_uploaded'],
                st.session_state['predictions'],
                st.session_state['probabilities'],
                encoder
            )

        with tabs[3]:
            render_explainability_view(
                st.session_state['df_uploaded'],
                st.session_state['predictions'],
                st.session_state['probabilities'],
                encoder,
                model,
                expected_columns,
                st.session_state['column_mapping']
            )

        with tabs[4]:
            render_detailed_results(
                st.session_state['df_uploaded'],
                st.session_state['predictions'],
                st.session_state['probabilities'],
                encoder,
                stats['risk_scores'],
                stats['risk_levels']
            )


if __name__ == "__main__":
    main()
