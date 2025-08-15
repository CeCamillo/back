import json
import io
from typing import Any, Dict, List, Union

import requests
import pandas as pd
import streamlit as st
import altair as alt


st.set_page_config(page_title="Network Anomaly Detection - Frontend", layout="wide")

# Sidebar configuration
st.sidebar.title("API Settings")
api_base_url: str = st.sidebar.text_input(
    label="API Base URL",
    value="http://127.0.0.1:5000",
    help="Base URL where FastAPI is running"
).rstrip("/")

if st.sidebar.button("Test connection"):
    try:
        # Ping feature importances as a lightweight GET
        resp = requests.get(f"{api_base_url}/features/importances", timeout=10)
        if resp.ok:
            st.sidebar.success("API reachable!")
        else:
            st.sidebar.error(f"API responded with status {resp.status_code}")
    except Exception as exc:
        st.sidebar.error(f"Connection failed: {exc}")


st.title("Network Anomaly Detection - Demo UI")
st.caption("Use this UI to try the FastAPI endpoints for JSON and CSV predictions and view model insights.")


def request_predict_json(base_url: str, payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> requests.Response:
    url = f"{base_url}/predict"
    headers = {"Content-Type": "application/json"}
    return requests.post(url, headers=headers, json=payload, timeout=60)


def request_predict_csv(base_url: str, file_bytes: bytes, filename: str) -> requests.Response:
    url = f"{base_url}/predict/csv"
    files = {"file": (filename, file_bytes, "text/csv")}
    return requests.post(url, files=files, timeout=120)


def request_feature_importances(base_url: str) -> List[Dict[str, Any]]:
    url = f"{base_url}/features/importances"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def request_predict_parquet(base_url: str, file_bytes: bytes, filename: str) -> requests.Response:
    url = f"{base_url}/predict/parquet"
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    return requests.post(url, files=files, timeout=120)


sample_single_json = {
    "dur": 0.000011,
    "proto": "udp",
    "service": "-",
    "state": "INT",
    "spkts": 2,
    "dpkts": 0,
    "sbytes": 104,
    "dbytes": 0,
    "rate": 90909.0902,
    "sttl": 254,
    "dttl": 0,
    "sload": 37818180.0,
    "dload": 0.0,
    "sloss": 0,
    "dloss": 0,
    "sinpkt": 0.000011,
    "dinpkt": 0.0,
    "sjit": 0.0,
    "djit": 0.0,
    "swin": 0,
    "stcpb": 0,
    "dtcpb": 0,
    "dwin": 0,
    "tcprtt": 0.0,
    "synack": 0.0,
    "ackdat": 0.0,
    "smean": 52,
    "dmean": 0,
    "trans_depth": 0,
    "response_body_len": 0,
    "ct_srv_src": 2,
    "ct_state_ttl": 2,
    "ct_dst_ltm": 1,
    "ct_src_dport_ltm": 1,
    "ct_dst_sport_ltm": 1,
    "ct_dst_src_ltm": 1,
    "is_ftp_login": 0,
    "ct_ftp_cmd": 0,
    "ct_flw_http_mthd": 0,
    "ct_src_ltm": 1,
    "ct_srv_dst": 2,
    "is_sm_ips_ports": 0
}


tabs = st.tabs(["Predict (JSON)", "Predict (CSV)", "Predict (Parquet)", "Feature Importances", "Batch Analysis", "Explore Data", "About"])

# Tab 1: JSON Predictions
with tabs[0]:
    st.subheader("Send JSON to /predict")
    st.caption("Paste a single JSON object or a JSON array of objects. Click Predict.")

    default_text = json.dumps(sample_single_json, indent=2)
    json_text = st.text_area("JSON input", value=default_text, height=280)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        predict_btn = st.button("Predict", type="primary")
    with col_b:
        st.write("")

    if predict_btn:
        try:
            payload = json.loads(json_text)
            resp = request_predict_json(api_base_url, payload)
            if resp.ok:
                data = resp.json()
                # The API can return a single object or a list
                if isinstance(data, dict):
                    st.success("Prediction OK")
                    result = data.get("result", {})
                    input_data = data.get("input_data", {})

                    st.metric("Prediction", result.get("prediction", "-"))
                    st.write("Confidence:", result.get("confidence", 0.0))

                    probs = result.get("probabilities", {})
                    if probs:
                        probs_df = (
                            pd.DataFrame(list(probs.items()), columns=["class", "probability"]) \
                              .sort_values("probability", ascending=False)
                        )
                        st.bar_chart(probs_df.set_index("class"))

                    with st.expander("Raw response"):
                        st.json(data)
                    with st.expander("Input data"):
                        st.json(input_data)
                elif isinstance(data, list):
                    st.success(f"Received {len(data)} predictions")

                    # Build a table summary
                    table_rows: List[Dict[str, Any]] = []
                    for item in data:
                        result = item.get("result", {})
                        row = {
                            "prediction": result.get("prediction"),
                            "is_anomaly": result.get("is_anomaly"),
                            "confidence": result.get("confidence")
                        }
                        table_rows.append(row)

                    st.dataframe(pd.DataFrame(table_rows))

                    with st.expander("Raw response"):
                        st.json(data)
                else:
                    st.warning("Unexpected response format")
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                st.error(f"Request failed: {resp.status_code}\n{err}")
        except json.JSONDecodeError as je:
            st.error(f"Invalid JSON: {je}")
        except Exception as exc:
            st.error(f"Error: {exc}")

# Tab 2: CSV Predictions
with tabs[1]:
    st.subheader("Upload CSV to /predict/csv")
    st.caption("Upload a CSV file with columns matching the training features.")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"]) 
    if uploaded_file is not None:
        # Show a preview of the uploaded CSV
        try:
            uploaded_file.seek(0)
            csv_bytes = uploaded_file.read()
            preview_df = pd.read_csv(io.BytesIO(csv_bytes))
            st.write("Preview (first 10 rows):")
            st.dataframe(preview_df.head(10))

            if st.button("Predict CSV", type="primary"):
                resp = request_predict_csv(api_base_url, csv_bytes, uploaded_file.name)
                if resp.ok:
                    data: List[Dict[str, Any]] = resp.json()
                    st.success(f"Received {len(data)} predictions")

                    # Build a compact table
                    rows = []
                    for item in data:
                        result = item.get("result", {})
                        row = {
                            "prediction": result.get("prediction"),
                            "is_anomaly": result.get("is_anomaly"),
                            "confidence": result.get("confidence")
                        }
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows))

                    with st.expander("Raw response"):
                        st.json(data)
                else:
                    try:
                        err = resp.json()
                    except Exception:
                        err = resp.text
                    st.error(f"Request failed: {resp.status_code}\n{err}")
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")

# Tab 3: Parquet Predictions
with tabs[2]:
    st.subheader("Upload Parquet to /predict/parquet")
    st.caption("Upload a Parquet file with columns matching the training features.")

    uploaded_parquet = st.file_uploader("Choose Parquet file", type=["parquet", "parq"]) 
    if uploaded_parquet is not None:
        try:
            uploaded_parquet.seek(0)
            parquet_bytes = uploaded_parquet.read()
            preview_df = pd.read_parquet(io.BytesIO(parquet_bytes))
            st.write("Preview (first 10 rows):")
            st.dataframe(preview_df.head(10))

            if st.button("Predict Parquet", type="primary"):
                resp = request_predict_parquet(api_base_url, parquet_bytes, uploaded_parquet.name)
                if resp.ok:
                    data: List[Dict[str, Any]] = resp.json()
                    st.success(f"Received {len(data)} predictions")

                    rows = []
                    for item in data:
                        result = item.get("result", {})
                        row = {
                            "prediction": result.get("prediction"),
                            "is_anomaly": result.get("is_anomaly"),
                            "confidence": result.get("confidence")
                        }
                            
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows))

                    with st.expander("Raw response"):
                        st.json(data)
                else:
                    try:
                        err = resp.json()
                    except Exception:
                        err = resp.text
                    st.error(f"Request failed: {resp.status_code}\n{err}")
        except Exception as exc:
            st.error(f"Failed to read Parquet: {exc}")

# Tab 4: Feature Importances
with tabs[3]:
    st.subheader("Top Feature Importances (/features/importances)")
    try:
        fi = request_feature_importances(api_base_url)
        if isinstance(fi, list) and fi:
            fi_df = pd.DataFrame(fi)
            fi_df = fi_df.sort_values("importance", ascending=True)
            st.bar_chart(fi_df.set_index("feature"))
            with st.expander("Raw data"):
                st.json(fi)
        else:
            st.info("No feature importances returned.")
    except Exception as exc:
        st.error(f"Failed to fetch importances: {exc}")

# Tab 5: Batch Analysis (upload file -> predict -> charts)
with tabs[4]:
    st.subheader("Batch Analysis: Upload CSV or Parquet, predict and visualize")
    uploaded_any = st.file_uploader("Choose CSV or Parquet", type=["csv", "parquet", "parq"])
    if uploaded_any is not None:
        try:
            uploaded_any.seek(0)
            raw_bytes = uploaded_any.read()
            if uploaded_any.name.lower().endswith(".csv"):
                df_preview = pd.read_csv(io.BytesIO(raw_bytes))
                predict_func = request_predict_csv
            else:
                df_preview = pd.read_parquet(io.BytesIO(raw_bytes))
                predict_func = request_predict_parquet

            st.write("Preview (first 10 rows):")
            st.dataframe(df_preview.head(10))

            run_btn = st.button("Run Batch Prediction", type="primary")
            if run_btn:
                resp = predict_func(api_base_url, raw_bytes, uploaded_any.name)
                if resp.ok:
                    results: List[Dict[str, Any]] = resp.json()
                    pred_rows = []
                    for item in results:
                        res = item.get("result", {})
                        pred_rows.append({
                            "prediction": res.get("prediction"),
                            "is_anomaly": res.get("is_anomaly"),
                            "confidence": res.get("confidence")
                        })
                    pred_df = pd.DataFrame(pred_rows)

                    st.markdown("**Predictions (first 100)**")
                    st.dataframe(pred_df.head(100))

                    st.markdown("**Class distribution**")
                    counts = pred_df["prediction"].value_counts().rename_axis("class").reset_index(name="count")
                    chart = alt.Chart(counts).mark_bar().encode(x=alt.X("class:N", sort='-y'), y="count:Q")
                    st.altair_chart(chart, use_container_width=True)

                    st.markdown("**Confidence histogram**")
                    conf_chart = alt.Chart(pred_df).mark_bar().encode(
                        x=alt.X("confidence:Q", bin=alt.Bin(maxbins=30)),
                        y="count()"
                    )
                    st.altair_chart(conf_chart, use_container_width=True)

                    show_anom = st.checkbox("Show only anomalies")
                    if show_anom:
                        st.dataframe(pred_df[pred_df["is_anomaly"] == True])

                    with st.expander("Raw API response"):
                        st.json(results)
                else:
                    try:
                        err = resp.json()
                    except Exception:
                        err = resp.text
                    st.error(f"Request failed: {resp.status_code}\n{err}")
        except Exception as exc:
            st.error(f"Failed to process file: {exc}")

# Tab 6: Explore Data (basic EDA)
with tabs[5]:
    st.subheader("Explore Data: Upload CSV or Parquet for EDA")
    uploaded_eda = st.file_uploader("Choose CSV or Parquet for EDA", type=["csv", "parquet", "parq"], key="eda")
    if uploaded_eda is not None:
        try:
            uploaded_eda.seek(0)
            raw_bytes = uploaded_eda.read()
            if uploaded_eda.name.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(raw_bytes))
            else:
                df = pd.read_parquet(io.BytesIO(raw_bytes))

            st.write("Shape:", df.shape)
            st.dataframe(df.head(10))

            # Missingness
            miss = df.isna().mean().sort_values(ascending=False)
            if (miss > 0).any():
                st.markdown("**Missing values (fraction)**")
                miss_df = miss.reset_index()
                miss_df.columns = ["column", "missing_fraction"]
                miss_chart = alt.Chart(miss_df).mark_bar().encode(x=alt.X("missing_fraction:Q"), y=alt.Y("column:N", sort='-x'))
                st.altair_chart(miss_chart, use_container_width=True)

            # Numeric summary
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if num_cols:
                st.markdown("**Numeric summary (describe)**")
                st.dataframe(df[num_cols].describe().T)

                # Correlation heatmap
                st.markdown("**Correlation heatmap**")
                corr = df[num_cols].corr(numeric_only=True)
                corr_df = corr.reset_index().melt('index')
                corr_df.columns = ["feature_x", "feature_y", "corr"]
                heat = alt.Chart(corr_df).mark_rect().encode(
                    x=alt.X("feature_x:N"),
                    y=alt.Y("feature_y:N"),
                    color=alt.Color("corr:Q", scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                    tooltip=["feature_x", "feature_y", alt.Tooltip("corr:Q", format=".2f")]
                ).properties(height=500)
                st.altair_chart(heat, use_container_width=True)

                # Single-column histogram
                sel_col = st.selectbox("Select a numeric column for histogram", num_cols)
                if sel_col:
                    hist = alt.Chart(df).mark_bar().encode(
                        x=alt.X(f"{sel_col}:Q", bin=alt.Bin(maxbins=40)),
                        y="count()"
                    )
                    st.altair_chart(hist, use_container_width=True)
            else:
                st.info("No numeric columns detected for summary/correlation.")
        except Exception as exc:
            st.error(f"Failed to run EDA: {exc}")

# Tab 7: About
with tabs[6]:
    st.markdown(
        """
        This Streamlit app provides a simple frontend to interact with the FastAPI
        anomaly detection service. Configure the API URL in the sidebar and use the tabs to:
        
        - Send JSON (single or batch) to `/predict`
        - Upload a CSV to `/predict/csv`
        - Upload a Parquet to `/predict/parquet`
        - Visualize global feature importances from `/features/importances`
        - Run batch predictions with charts
        - Explore raw data (EDA)
        
        The app expects the API to be running locally by default at `http://127.0.0.1:5000`.
        """
    ) 