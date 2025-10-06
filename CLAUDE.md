# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Network Anomaly Detection Dashboard - An interactive Streamlit application that allows users to upload their network traffic data and detect anomalies (Normal, Backdoor, Worms) using a pre-trained machine learning model.

## Key Commands

### Run Anomaly Detection Dashboard
```bash
streamlit run anomaly_detector.py
```
Launches the interactive dashboard at http://localhost:8501 where users can upload network data and view detection results.

### Train Pre-trained Model (One-time Setup)
```bash
python model_training.py
```
Generates the pre-trained model artifacts required by the dashboard. Only needs to be run once or when retraining is desired.

### MLflow UI (Optional - for model development)
```bash
mlflow ui --backend-store-uri mlruns --port 5001
```
Opens MLflow UI at http://127.0.0.1:5001 to inspect training experiments.

## Architecture

### Main Application ([anomaly_detector.py](anomaly_detector.py))

**Core workflow:**
1. User uploads CSV/Parquet network traffic file
2. System auto-suggests column mapping to UNSW-NB15 format
3. User confirms/adjusts mapping via interactive UI
4. Data is preprocessed and fed to pre-trained model
5. Results displayed in three tabs: Detection Results, Network Insights, Feature Importance

**Column Mapping System** (`suggest_column_mapping`):
- Performs case-insensitive exact matching first
- Falls back to partial string matching (handles underscores, variations)
- User can manually override any auto-suggestion via selectboxes
- Missing columns filled with defaults (0 for numeric, 'unknown' for categorical)

**Preprocessing Pipeline** (`preprocess_uploaded_data`):
1. Renames user columns based on mapping
2. Fills missing UNSW columns with defaults
3. One-hot encodes categorical features (proto, service, state)
4. Reindexes to match training columns exactly using `model_columns.joblib`

**Model Loading** (`load_model_artifacts`):
- Loads latest `best_model_pipeline_*.joblib` (StandardScaler + Classifier)
- Loads `label_encoder.joblib` for class names
- Loads `model_columns.joblib` for feature alignment
- Cached with `@st.cache_resource` for performance

**UI Sections:**

1. **Detection Results Tab** (`render_detection_results`):
   - Shows all records with prediction and confidence score
   - Summary metrics (total, normal, backdoor, worms counts)
   - Interactive filters: by class, by minimum confidence
   - Download button for filtered results as CSV

2. **Network Insights Tab** (`render_insights`):
   - Anomaly rate and distribution pie chart
   - Protocol analysis grouped by prediction class
   - Top services by volume
   - Average byte/packet transfer analysis
   - All visualizations adapt to available columns

3. **Feature Importance Tab** (`render_feature_importance`):
   - Bar chart of top 20 most important features
   - Expandable table with all features ranked
   - Works with tree-based models (RF, XGBoost, LightGBM)

### Training Script ([model_training.py](model_training.py))

**Purpose**: Generate pre-trained model artifacts for the dashboard (one-time setup)

**Data preprocessing** (`preprocess_data` function):
1. Filters dataset to Normal (label=0), Worms, and Backdoor attacks only
2. Creates `attack_label` column from `attack_cat` (fillna 'Normal')
3. Drops metadata columns: `id`, `label`, `attack_cat`
4. Returns X (features) and y (attack_label)

**Feature engineering**:
- One-hot encoding via `pd.get_dummies()` on all categorical features
- Train/test alignment using `align(join="left")` to ensure consistent columns
- StandardScaler in pipeline before classifier

**Model training loop**:
- Pipeline: StandardScaler → Classifier
- RandomizedSearchCV with `f1_weighted` scoring, 10 iterations, 3-fold CV
- Three models with class balancing:
  - RandomForest (class_weight='balanced')
  - XGBoost (eval_metric='mlogloss')
  - LightGBM (class_weight='balanced')

**MLflow logging**:
- Experiment name: "UNSW_NB15_Classification"
- Per model run logs: best params, f1_weighted, per-class metrics, confusion matrix PNG, sklearn model
- Global best model selected by highest f1_weighted

**Artifacts generated** (required by anomaly_detector.py):
- `best_model_pipeline_<ALG>.joblib` - Full pipeline (scaler + model)
- `model_columns.joblib` - Column index from training for inference alignment
- `label_encoder.joblib` - LabelEncoder with class mapping
- `confusion_matrix_<ALG>.png` - Visualizations (optional)
- `mlruns/` - MLflow experiment tracking (optional)

## Expected Column Format

The dashboard expects UNSW-NB15 format columns (43 features):

**Flow statistics**: dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, dwin, stcpb, dtcpb, tcprtt, synack, ackdat, smean, dmean

**Connection features**: trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, ct_src_ltm, ct_srv_dst, ct_flw_http_mthd

**Categorical**: proto (protocol), service, state

**Flags**: is_ftp_login, is_sm_ips_ports, ct_ftp_cmd

**Note**: Users don't need exact column names - the mapping interface handles variations and missing columns

## Important Implementation Details

### Session State Management
Detection results stored in `st.session_state`:
- `predictions` - numpy array of class predictions
- `probabilities` - numpy array of class probabilities
- `df_uploaded` - original uploaded dataframe
- `column_mapping` - user's column mapping dict

Persists results across tab switches without re-running detection.

### Flexible Data Handling
- Auto-fills missing columns to allow partial datasets
- Handles unknown categorical values via one-hot encoding
- Gracefully adapts insights to available columns
- No strict validation - best-effort approach

### Feature Alignment
Critical for inference: uploaded data must be reindexed to match training columns exactly using `model_columns.joblib`. Uses `X.reindex(columns=expected_columns, fill_value=0)` in `preprocess_uploaded_data`.

### Class Filtering
Both training and detection focus on 3 classes:
- Normal (label=0)
- Worms (attack_cat='Worms')
- Backdoor (attack_cat='Backdoor')

### Model Requirements
Dashboard requires all three artifacts:
- Pipeline (contains both preprocessing and model)
- Label encoder (for class names)
- Column index (for feature alignment)

Missing artifacts show helpful error with setup instructions.

## Dependencies

Core libraries (see [requirements.txt](requirements.txt)):
- **ML**: scikit-learn, xgboost, lightgbm, mlflow, joblib
- **Data**: pandas, numpy, pyarrow
- **Viz**: matplotlib, seaborn, streamlit
- **API** (for future enhancement): fastapi, uvicorn

## File Organization

**Main application**:
- `anomaly_detector.py` - Primary Streamlit dashboard for user uploads

**Training (one-time setup)**:
- `model_training.py` - Generates pre-trained model artifacts
- `datasets/` - Training parquet files (UNSW-NB15)

**Legacy files** (can be removed):
- `streamlit_training_dashboard.py` - Old training results viewer

**Generated artifacts** (after training):
- `best_model_pipeline_*.joblib`
- `model_columns.joblib`
- `label_encoder.joblib`
- `mlruns/` - MLflow tracking data (optional)

## Development Notes

### Adding New Insights
To add new network insights in `analyze_network_behavior`:
1. Check if required columns exist in `column_mapping`
2. Verify columns present in uploaded dataframe
3. Add analysis to `insights` dict
4. Render in `render_insights` function

### Supporting New Column Types
To add new expected columns:
1. Add to `UNSW_CRITICAL_COLUMNS` list in anomaly_detector.py
2. Specify default value in `preprocess_uploaded_data` (0 for numeric, 'unknown' for categorical)
3. Column mapper will auto-suggest matches

### Model Retraining
To use a different model:
1. Modify `model_training.py` with new algorithm/params
2. Run training to generate new artifacts
3. Dashboard automatically loads latest model by timestamp
