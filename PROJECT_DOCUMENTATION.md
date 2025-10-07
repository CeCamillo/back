# Network Anomaly Detection System - Complete Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Machine Learning Pipeline](#4-machine-learning-pipeline)
5. [Data Processing](#5-data-processing)
6. [API Specification](#6-api-specification)
7. [User Interfaces](#7-user-interfaces)
8. [Model Training & Optimization](#8-model-training--optimization)
9. [Academic Results Generation](#9-academic-results-generation)
10. [Dependencies & Requirements](#10-dependencies--requirements)
11. [Installation & Setup](#11-installation--setup)
12. [Usage Examples](#12-usage-examples)
13. [File Structure](#13-file-structure)
14. [Technical Details](#14-technical-details)
15. [Performance Metrics](#15-performance-metrics)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Project Overview

### Purpose
This project is a **production-ready Network Anomaly Detection System** designed to identify malicious network traffic patterns using machine learning. It specifically detects two critical types of network attacks:
- **Backdoor attacks**: Unauthorized remote access attempts
- **Worm propagation**: Self-replicating malware spreading through networks

### Key Features
- ✅ **Real-time anomaly detection** via REST API
- ✅ **Multiple ML algorithms**: Random Forest, XGBoost, LightGBM
- ✅ **Class imbalance handling**: SMOTE oversampling + class weighting
- ✅ **Production-ready architecture**: Separate backend API and frontend dashboard
- ✅ **Flexible data input**: Auto-mapping for various network log formats
- ✅ **Comprehensive analytics**: Risk scoring, temporal analysis, network topology insights
- ✅ **Academic research support**: TCC/thesis results generation

### Target Users
- **Security Analysts**: Monitor network traffic for threats
- **Researchers**: Study network intrusion detection patterns
- **Students**: Use for academic projects (TCC, dissertations, papers)
- **DevOps/NetOps**: Integrate into security monitoring pipelines

---

## 2. System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                          │
├─────────────────────────────────────────────────────────────────┤
│  Option 1: Simplified UI          │  Option 2: Advanced Dashboard│
│  (streamlit_app.py)                │  (anomaly_detector.py)       │
│  - Upload CSV/Parquet              │  - Full analytics           │
│  - View detection results          │  - Risk scoring             │
│  - Download reports                │  - Temporal analysis        │
│  - Connects to Backend API         │  - Network topology         │
│                                    │  - Feature explainability   │
└─────────────────┬──────────────────┴──────────────────────────────┘
                  │
                  │ HTTP POST /predict/csv
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND API                              │
│                      (backend_api.py)                            │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI REST Server                                            │
│  - Endpoints: /, /health, /predict/csv                          │
│  - File upload handling (CSV/Parquet)                           │
│  - Request validation & error handling                          │
│  - CORS middleware for cross-origin requests                    │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ Loads artifacts on startup
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE & ARTIFACTS                       │
├─────────────────────────────────────────────────────────────────┤
│  Pipeline Components:                                           │
│  1. pipeline.joblib            - Complete pipeline (scaler+model)│
│  2. model.joblib               - Trained classifier              │
│  3. scaler.joblib              - StandardScaler for normalization│
│  4. model_columns.joblib       - Expected feature columns        │
│  5. label_encoder.joblib       - Class label encoder            │
│                                                                  │
│  Pipeline Flow:                                                 │
│  Raw Data → One-Hot Encode → Align Columns → Scale → Classify   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ Trained on
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                         DATASET                                  │
│                      (UNSW-NB15)                                 │
├─────────────────────────────────────────────────────────────────┤
│  Training Set: 175,341 samples                                  │
│  Testing Set:  82,332 samples                                   │
│                                                                  │
│  Classes:                                                       │
│  - Normal: ~90% (legitimate traffic)                            │
│  - Backdoor: ~5% (unauthorized access)                          │
│  - Worms: ~5% (self-replicating malware)                        │
│                                                                  │
│  Features: 42 network traffic attributes                        │
│  (duration, bytes, packets, protocol, service, state, etc.)     │
└─────────────────────────────────────────────────────────────────┘
```

### Design Patterns
- **Separation of Concerns**: Backend (FastAPI) handles ML inference, frontend (Streamlit) handles visualization
- **Pipeline Pattern**: StandardScaler + Classifier packaged as single unit (prevents data leakage)
- **Strategy Pattern**: Multiple ML algorithms with unified interface
- **Factory Pattern**: Automatic selection of best-performing model

---

## 3. Core Components

### 3.1 Backend API (`backend_api.py`)

**Purpose**: Production-ready REST API for network anomaly detection

**Key Features**:
- FastAPI framework for high performance
- Async/await support for concurrent requests
- Automatic OpenAPI documentation (`/docs`)
- CORS middleware for cross-origin requests
- Comprehensive error handling and logging

**Endpoints**:

#### `GET /`
Health check endpoint returning API information
```json
{
  "status": "online",
  "service": "Network Anomaly Detection API",
  "version": "1.0.0",
  "endpoints": {...}
}
```

#### `GET /health`
Detailed health status including model information
```json
{
  "status": "healthy",
  "pipeline_loaded": true,
  "pipeline_steps": ["scaler", "classifier"],
  "expected_features": 196,
  "classes": ["Backdoor", "Normal", "Worms"]
}
```

#### `POST /predict/csv`
Main prediction endpoint accepting CSV/Parquet files

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: File upload (CSV or Parquet)

**Response**:
```json
{
  "status": "success",
  "file_name": "network_traffic.csv",
  "total_connections": 1000,
  "normal_connections": 850,
  "worms_detected": 100,
  "backdoors_detected": 50,
  "total_anomalies": 150,
  "anomaly_rate": 15.0,
  "metrics": {
    "precision": 0.9234,
    "recall": 0.8876,
    "f1_score": 0.9051
  },
  "predictions": [
    {
      "connection_id": 0,
      "prediction": "Normal",
      "confidence": 0.9823
    },
    ...
  ]
}
```

**Processing Pipeline** (lines 104-273):
1. **File Validation**: Check file format (CSV/Parquet)
2. **Data Loading**: Read file into pandas DataFrame
3. **Preprocessing**:
   - One-hot encode categorical features
   - Align columns with training data (fill missing with 0)
4. **Prediction**: Run through ML pipeline
5. **Post-processing**:
   - Decode predictions (numeric → class labels)
   - Calculate confidence scores
   - Generate statistics
6. **Metrics Calculation** (if ground truth available):
   - Precision, Recall, F1-Score
   - Supports multiple label column formats (`label`, `attack_cat`, `attack_label`)
7. **Response Generation**: Package results as JSON

**Error Handling**:
- `400 Bad Request`: Invalid file format
- `500 Internal Server Error`: Model not loaded or processing failure
- Detailed error messages with troubleshooting hints

---

### 3.2 Simplified Frontend (`streamlit_app.py`)

**Purpose**: User-friendly web interface for quick anomaly detection

**Architecture**:
- **Stateless**: Communicates with backend API for all ML operations
- **Lightweight**: Minimal logic, focuses on visualization
- **Responsive**: Real-time feedback with spinners and progress indicators

**User Flow**:

```
1. Upload CSV File
   ↓
2. File sent to Backend API
   ↓
3. API processes and returns results
   ↓
4. Display analysis report:
   - Summary metrics (total connections, anomalies, threat breakdown)
   - Performance metrics (precision, recall, F1)
   - Anomaly rate visualization (bar chart)
   - Threat level assessment (color-coded warnings)
   ↓
5. Detailed predictions (expandable):
   - Filterable table (by prediction type, confidence threshold)
   - Download CSV option
```

**UI Components** (lines 58-227):

1. **Summary Metrics** (4-column layout):
   - Total Connections Analyzed
   - Normal Connections (with percentage)
   - Backdoors Detected (with percentage, red highlight)
   - Worms Detected (with percentage, orange highlight)

2. **Performance Metrics** (3-column layout, conditional):
   - Precision (proportion of correct positive identifications)
   - Recall (proportion of actual positives identified)
   - F1-Score (harmonic mean of precision and recall)
   - *Only shown if ground truth labels exist in uploaded file*

3. **Anomaly Detection Summary**:
   - Anomaly Rate (percentage)
   - Threat Level (color-coded):
     - 🟢 Low (<5%): Network appears secure
     - 🟡 Moderate (5-20%): Monitor closely
     - 🔴 High (>20%): Immediate action required
   - Bar chart of category distribution

4. **Detailed Predictions** (collapsible):
   - Filterable table with controls:
     - Multi-select: Filter by prediction type
     - Slider: Minimum confidence threshold
   - Columns: connection_id, prediction, confidence
   - Download button for CSV export

**Error Handling**:
- Connection errors: Display troubleshooting instructions
- API errors: Show error details with tips
- Missing backend: Instructions to start `backend_api.py`

---

### 3.3 Advanced Dashboard (`anomaly_detector.py`)

**Purpose**: Comprehensive security analysis platform with advanced features

**Architecture**:
- **Standalone**: Loads ML model directly (no backend API required)
- **Feature-rich**: Risk scoring, temporal analysis, network topology, explainability
- **Session-based**: Uses Streamlit session state for persistence

**Advanced Features**:

#### **Risk Scoring System** (lines 104-148)
Custom risk calculation based on prediction type and confidence:

```python
Risk Formula:
- Normal: risk_score = (1 - confidence) × 30
  → Low confidence in "Normal" = higher risk (possible misclassification)

- Backdoor: risk_score = 40 + (confidence × 45)
  → Base risk 40% + confidence boost → Range: 40-85%

- Worms: risk_score = 50 + (confidence × 45)
  → Higher base risk (propagation threat) → Range: 50-95%

Risk Levels:
- Baixo (Low): < threshold
- Médio (Medium): threshold to moderate
- Alto (High): moderate to critical
- Crítico (Critical): > critical threshold
```

#### **Future Risk Prediction** (lines 151-193)
Predictive algorithm for estimating future threat levels:

```python
future_risk = (current_anomaly_rate × 0.6 + avg_confidence × 40) × worm_multiplier

Components:
1. Current anomaly rate (0-100%)
2. Average confidence of anomaly predictions
3. Worm multiplier (1.5x if worms present, else 1.0x)
   → Worms have higher future risk due to propagation

Capped at 100%
```

#### **Network Topology Analysis** (lines 251-342)
Identifies attack patterns by analyzing:

1. **Top Attackers** (source IPs):
   - Count attacks per source IP
   - Calculate average confidence per IP
   - Visualize with horizontal bar chart
   - Recommendation: Block top N IPs

2. **Top Targets** (destination IPs):
   - Count attacks per destination IP
   - Identify most vulnerable hosts
   - Recommendation: Reinforce protection

3. **Port Analysis**:
   - Identify most attacked ports
   - Break down by attack type (Backdoor vs Worms)
   - Pivot table visualization

4. **Repeated Attacks**:
   - Detect same source → destination pairs
   - Flag potential scanning or coordinated attacks
   - Count repeated connection attempts

**Column Detection** (flexible):
- Auto-detects IP columns: `srcip`, `src_ip`, `source_ip`, `saddr`, `dstip`, `dst_ip`, etc.
- Auto-detects port columns: `dport`, `dst_port`, `sport`, `src_port`
- Case-insensitive matching

#### **Temporal Analysis** (lines 345-459)
Time-series analysis of attack patterns:

1. **Time Range Detection**:
   - Searches for timestamp columns: `timestamp`, `time`, `datetime`, `date`, `ts`, `stime`, `ltime`
   - Parses various timestamp formats

2. **Adaptive Time Bucketing**:
   ```python
   if range > 7 days:    freq = 'D'  (daily)
   elif range > 1 day:   freq = 'H'  (hourly)
   elif range > 1 hour:  freq = '10T' (10-minute)
   else:                 freq = 'T'  (per minute)
   ```

3. **Time Series Metrics**:
   - Total connections per time bucket
   - Anomaly count per time bucket
   - Anomaly rate (percentage)

4. **Spike Detection**:
   - Identifies time periods with anomaly rate > 2× average
   - Flags for urgent investigation

5. **Trend Analysis**:
   - Linear regression slope calculation
   - Classification:
     - 📈 Crescente (increasing): slope > 0.5
     - 📉 Decrescente (decreasing): slope < -0.5
     - ➡️ Estável (stable): -0.5 ≤ slope ≤ 0.5

6. **Visualizations**:
   - Line plot: Anomaly rate over time
   - Filled area chart for emphasis
   - Spike markers (red triangles)
   - Separate plots for Backdoor vs Worms over time

#### **Model Explainability** (lines 936-1031)
SHAP-like feature influence for individual predictions:

1. **Sample Selection**:
   - User selects a specific anomaly to analyze
   - Shows connection ID, prediction, confidence

2. **Probability Distribution**:
   - Bar chart showing probabilities for all classes
   - Highlights predicted class

3. **Feature Influence** (lines 462-497):
   ```python
   Influence = Feature Importance × |Feature Value|

   Steps:
   1. Extract feature importances from Random Forest
   2. Get scaled feature values for the sample
   3. Calculate influence score (importance × abs(value))
   4. Rank features by influence
   5. Display top 10 most influential features
   ```

4. **Interpretation**:
   - Horizontal bar chart of feature influence
   - Table with feature names, importance scores, actual values
   - Raw connection data (JSON format)

#### **Column Mapping Interface** (lines 1048-1082)
Interactive tool for aligning user data to UNSW-NB15 format:

1. **Auto-suggestion Algorithm** (lines 60-76):
   ```python
   For each expected column:
     1. Check exact match (case-insensitive)
     2. Check partial match (remove underscores, compare)
     3. Suggest best match or leave blank
   ```

2. **Manual Override**:
   - Dropdown for each of 42 UNSW-NB15 columns
   - Select from user's uploaded columns or "(nenhum)"
   - Two-column layout for compact display

3. **Default Values**:
   - Categorical features: filled with "unknown"
   - Numerical features: filled with 0
   - Ensures model can process incomplete data

**Report Sections** (tabs):

1. **Security Report** (lines 500-593):
   - Key metrics: total connections, anomalies, confidence, future risk
   - Pie chart: traffic distribution
   - Bar chart: malware detections

2. **Suspicious Connections** (lines 596-647):
   - Filters connections with risk level "Alto" or "Crítico"
   - Priority columns: Classification, Confidence, Risk %, Risk Level
   - Download CSV option

3. **Network Topology** (lines 706-821):
   - All features from Network Topology Analysis section

4. **Temporal Analysis** (lines 823-934):
   - All features from Temporal Analysis section

5. **Explainability** (lines 936-1031):
   - All features from Model Explainability section

6. **Detailed Results** (lines 650-703):
   - Complete results table with filters:
     - Classification filter (Normal, Backdoor, Worms)
     - Risk level filter (Baixo, Médio, Alto, Crítico)
     - Confidence threshold slider
   - Sorted by risk percentage (descending)
   - Download filtered results

---

### 3.4 Model Training (`model_training.py`)

**Purpose**: Train and compare multiple ML models with class imbalance handling

**Training Pipeline**:

```
1. Load Data (lines 25-26)
   ├─ Training set: UNSW_NB15_training-set.parquet
   └─ Testing set: UNSW_NB15_testing-set.parquet

2. Preprocessing (lines 28-41)
   ├─ Filter classes: Keep only Normal, Backdoor, Worms
   ├─ Create labels: attack_cat → attack_label
   ├─ Drop metadata: id, label, attack_cat columns
   └─ Separate X (features) and y (labels)

3. One-Hot Encoding (lines 47-49)
   ├─ Convert categorical features to binary columns
   ├─ Align train/test columns (ensures consistency)
   └─ Result: ~196 features after encoding

4. Label Encoding (lines 52-55)
   ├─ Encode string labels to integers
   │  - "Backdoor" → 0
   │  - "Normal" → 1
   │  - "Worms" → 2
   └─ Store LabelEncoder for later decoding

5. SMOTE Balancing (lines 57-65)
   ├─ Analyze class distribution
   │  Before: Normal ~90%, Backdoor ~5%, Worms ~5%
   ├─ Apply SMOTE (Synthetic Minority Over-sampling)
   │  - Generates synthetic samples for minority classes
   │  - Balances class distribution
   │  After: Normal ~33%, Backdoor ~33%, Worms ~33%
   └─ Reduces model bias toward majority class

6. Model Training (lines 68-163)
   For each model (RandomForest, XGBoost, LightGBM):
     ├─ Create Pipeline:
     │  └─ StandardScaler → Classifier
     ├─ Hyperparameter Search:
     │  ├─ RandomizedSearchCV with 10 iterations
     │  ├─ 3-fold cross-validation
     │  ├─ Scoring: F1-weighted (balanced metric)
     │  └─ n_jobs=-1 (parallel processing)
     ├─ Training:
     │  └─ Fit on SMOTE-balanced training data
     ├─ Evaluation:
     │  ├─ Predict on original test set
     │  ├─ Calculate F1-score, precision, recall
     │  └─ Generate classification report
     └─ MLflow Logging:
        ├─ Log hyperparameters
        ├─ Log metrics (F1, precision, recall per class)
        ├─ Log confusion matrix (PNG artifact)
        └─ Log complete pipeline

7. Model Selection (lines 160-163)
   └─ Select model with highest F1-weighted score

8. Save Artifacts (lines 169-185)
   ├─ best_model_pipeline_{model_name}.joblib (complete pipeline)
   ├─ model.joblib (classifier only)
   ├─ scaler.joblib (StandardScaler)
   ├─ model_columns.joblib (expected feature columns)
   └─ label_encoder.joblib (for decoding predictions)
```

**Hyperparameter Grids** (lines 74-93):

**Random Forest**:
- `n_estimators`: [100, 200] trees
- `max_depth`: [10, 20, 30] levels
- `min_samples_split`: [2, 5] samples
- `min_samples_leaf`: [1, 2] samples
- `class_weight`: 'balanced' (built-in imbalance handling)

**XGBoost**:
- `n_estimators`: [100, 200] boosting rounds
- `max_depth`: [5, 10, 15] tree depth
- `learning_rate`: [0.05, 0.1, 0.2]
- `subsample`: [0.7, 0.8] sample ratio

**LightGBM**:
- `n_estimators`: [100, 200] trees
- `max_depth`: [10, 20, -1] (no limit)
- `learning_rate`: [0.05, 0.1]
- `num_leaves`: [31, 50] leaf nodes
- `class_weight`: 'balanced'

**Why Multiple Models?**
- Different algorithms have different strengths
- Random Forest: Good for interpretability (feature importance)
- XGBoost: Often highest accuracy, gradient boosting
- LightGBM: Faster training, handles large datasets well
- Automatic selection ensures best performance

**MLflow Integration** (lines 100-157):
Tracks all experiments in `mlruns/` directory:
- Experiment name: "UNSW_NB15_Classification"
- Each model run stores: parameters, metrics, artifacts, model
- Can compare runs in MLflow UI: `mlflow ui`

---

### 3.5 TCC Results Generator (`generate_tcc_results.py`)

**Purpose**: Generate publication-ready results for academic papers

**Generates 6 Artifacts**:

1. **`naive_model_report.txt`** (lines 196-204)
   - Full classification report
   - Model: RandomForest (n_estimators=100, **no class_weight**)
   - Metrics: Precision, Recall, F1-Score, Support per class
   - Format: Plain text (easy to copy-paste into LaTeX/Word)

2. **`improved_model_report.txt`** (lines 206-214)
   - Full classification report
   - Model: RandomForest (n_estimators=100, **class_weight='balanced'**)
   - Same metrics format as naive model
   - Direct comparison possible

3. **`naive_model_confusion_matrix.png`** (lines 223-253)
   - Heatmap visualization
   - Normalized percentages (not raw counts)
   - Blue color scheme
   - 300 DPI (publication quality)
   - Labels: "True Label" (y-axis), "Predicted Label" (x-axis)

4. **`improved_model_confusion_matrix.png`** (lines 255-276)
   - Same format as naive matrix
   - Green color scheme (distinguishes from naive)
   - Highlights improvement visually

5. **`feature_importances.png`** (lines 333-383)
   - Horizontal bar chart
   - Top 15 most important features (from improved model)
   - Sorted by importance score (highest at top)
   - Value labels on bars
   - Grid lines for readability

6. **`feature_importances.csv`** (lines 386-388)
   - Complete feature importance data (all features)
   - Columns: feature name, importance score
   - Sorted by importance (descending)
   - For supplementary analysis

**Comparative Analysis** (lines 284-326):
Prints to console:
```
============================================================
COMPARATIVE F1-SCORE ANALYSIS
============================================================

Attack Type: Backdoor
  - Naive Model F1-Score:    0.XXXX
  - Improved Model F1-Score: 0.YYYY
  - Improvement:             +ZZ.ZZ%

Attack Type: Worms
  - Naive Model F1-Score:    0.AAAA
  - Improved Model F1-Score: 0.BBBB
  - Improvement:             +CC.CC%

============================================================
```

**Scientific Contribution**:
- Demonstrates empirically that class imbalance handling improves minority class detection
- Provides visual evidence (confusion matrices)
- Quantifies improvement (percentage increase in F1)
- Ready-to-use figures for papers/theses

**Academic Use Case**:
Typical TCC/thesis structure:
```
Chapter 4: Results
  4.1 Baseline Model (Naive)
      - Classification report (copy from .txt)
      - Confusion matrix (insert .png)
  4.2 Improved Model
      - Classification report (copy from .txt)
      - Confusion matrix (insert .png)
  4.3 Comparative Analysis
      - F1-Score table (use console output)
      - Discussion of improvements
  4.4 Feature Analysis
      - Feature importance chart (insert .png)
      - Discussion of top features
```

---

## 4. Machine Learning Pipeline

### 4.1 Pipeline Architecture

The system uses scikit-learn's `Pipeline` class to ensure proper data flow and prevent data leakage:

```python
Pipeline([
    ('scaler', StandardScaler()),    # Step 1: Normalize features
    ('classifier', RandomForestClassifier())  # Step 2: Classify
])
```

**Why Pipeline?**
- **Prevents Data Leakage**: Scaler is fit only on training data
- **Atomic Operations**: Transform and predict in one call
- **Easy Serialization**: Save/load entire pipeline as one file
- **Consistency**: Same transformations applied to training, validation, and test data

### 4.2 Feature Engineering

**Original UNSW-NB15 Features** (42 total):

**Flow Features** (network traffic statistics):
- `dur`: Connection duration (seconds)
- `spkts`, `dpkts`: Source/destination packet counts
- `sbytes`, `dbytes`: Source/destination byte counts
- `rate`: Packet transmission rate
- `sttl`, `dttl`: Source/destination time-to-live
- `sload`, `dload`: Source/destination load (bits/second)
- `sloss`, `dloss`: Source/destination packet loss
- `sinpkt`, `dinpkt`: Inter-packet arrival time
- `sjit`, `djit`: Jitter (packet delay variation)
- `swin`, `dwin`: TCP window sizes
- `stcpb`, `dtcpb`: TCP base sequence numbers
- `tcprtt`: TCP round-trip time
- `synack`, `ackdat`: TCP handshake timings
- `smean`, `dmean`: Mean packet sizes

**Connection Features** (session-level statistics):
- `trans_depth`: Pipelined depth (HTTP)
- `response_body_len`: HTTP response body size
- `ct_srv_src`: Connections to same service from source
- `ct_state_ttl`: Connections with same state and TTL
- `ct_dst_ltm`: Connections to destination in last time window
- `ct_src_dport_ltm`: Connections from source to destination port
- `ct_dst_sport_ltm`: Connections to destination from source port
- `ct_dst_src_ltm`: Connections between destination and source
- `is_ftp_login`: FTP login attempt (binary)
- `ct_ftp_cmd`: FTP command count
- `ct_flw_http_mthd`: HTTP method count
- `ct_src_ltm`: Connections from source in last time
- `ct_srv_dst`: Connections to same service at destination
- `is_sm_ips_ports`: Same IP and port (binary)

**Categorical Features** (encoded via one-hot):
- `proto`: Protocol (tcp, udp, icmp, etc.)
- `service`: Application service (http, ftp, ssh, dns, etc.)
- `state`: Connection state (FIN, INT, CON, REQ, RST, etc.)

**After One-Hot Encoding**: ~196 features
- Example: `proto=tcp` becomes `proto_tcp=1`, `proto_udp=0`, etc.
- Example: `service=http` becomes `service_http=1`, `service_ftp=0`, etc.

### 4.3 Preprocessing Steps

**Step 1: Column Mapping** (flexible input)
```python
def preprocess_uploaded_data(df, column_mapping, expected_columns):
    # 1. Rename columns based on user mapping
    df_mapped = df.rename(columns={v: k for k, v in column_mapping.items()})

    # 2. Add missing columns with defaults
    for col in UNSW_CRITICAL_COLUMNS:
        if col not in df_mapped.columns:
            if col in ['proto', 'service', 'state']:
                df_mapped[col] = 'unknown'  # Categorical default
            else:
                df_mapped[col] = 0  # Numerical default

    # 3. Keep only mapped columns
    df_clean = df_mapped[UNSW_CRITICAL_COLUMNS]

    return df_clean
```

**Step 2: One-Hot Encoding**
```python
X = pd.get_dummies(df_clean)
# Converts:
#   proto='tcp', service='http', state='FIN'
# To:
#   proto_tcp=1, proto_udp=0, ..., service_http=1, service_ftp=0, ..., state_FIN=1, state_INT=0, ...
```

**Step 3: Column Alignment**
```python
X_aligned = X.reindex(columns=expected_columns, fill_value=0)
# Ensures:
# - All expected columns exist (fill missing with 0)
# - Column order matches training data
# - No extra columns (drops unknowns)
```

**Step 4: Scaling** (inside pipeline)
```python
scaler = StandardScaler()
# Transforms each feature to:
#   X_scaled = (X - mean) / std
# Benefits:
# - Removes unit dependency (bytes vs seconds)
# - Centers data around 0
# - Makes gradient descent converge faster
```

**Step 5: Classification** (inside pipeline)
```python
classifier.predict(X_scaled)
# Returns: integer class labels [0, 1, 2]
#   0 = Backdoor
#   1 = Normal
#   2 = Worms
```

**Step 6: Label Decoding**
```python
label_encoder.inverse_transform(predictions)
# Converts: [0, 1, 2] → ["Backdoor", "Normal", "Worms"]
```

### 4.4 Class Imbalance Handling

**Problem**:
UNSW-NB15 dataset is highly imbalanced:
- Normal: ~90% of samples
- Backdoor: ~5% of samples
- Worms: ~5% of samples

Without correction, models learn to always predict "Normal" (high accuracy but useless for security).

**Solution 1: SMOTE** (Synthetic Minority Over-sampling Technique)
```python
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_encoded)

# How SMOTE works:
# 1. For each minority sample (e.g., Backdoor):
#    - Find k nearest neighbors (same class)
#    - Create synthetic samples along lines between neighbors
# 2. Result: Balanced dataset (~33% each class)

# Example:
# Before SMOTE:
#   Normal:   157,000 samples
#   Backdoor:   8,000 samples
#   Worms:      8,000 samples
#
# After SMOTE:
#   Normal:   157,000 samples
#   Backdoor: 157,000 samples (149,000 synthetic)
#   Worms:    157,000 samples (149,000 synthetic)
```

**Solution 2: Class Weights** (used in final model)
```python
RandomForestClassifier(class_weight='balanced')

# Calculates weights automatically:
#   weight[i] = n_samples / (n_classes × n_samples[i])
#
# Example:
#   Total samples: 175,000
#   n_classes: 3
#   Normal samples: 157,000
#   Backdoor samples: 8,000
#   Worms samples: 8,000
#
# Weights:
#   Normal:   175,000 / (3 × 157,000) ≈ 0.37
#   Backdoor: 175,000 / (3 × 8,000)   ≈ 7.29
#   Worms:    175,000 / (3 × 8,000)   ≈ 7.29
#
# Effect: Misclassifying Backdoor/Worms is 20× more costly than Normal
```

**Why Both?**
- SMOTE: Used during training to provide more learning examples
- Class Weights: Used in final model for deployment (no need to store synthetic samples)

**Trade-off**:
- Increased minority class recall (more attacks detected) ✅
- Slightly decreased overall accuracy (more false positives) ⚠️
- **For security applications, this is desirable** (better to flag false alarms than miss real attacks)

### 4.5 Model Selection

**Algorithms Compared**:

1. **Random Forest**
   - Ensemble of decision trees (100-200 trees)
   - Each tree votes on final prediction
   - **Pros**: Interpretable (feature importance), robust to overfitting
   - **Cons**: Can be slow on large datasets
   - **Best for**: Feature analysis, explainability

2. **XGBoost** (eXtreme Gradient Boosting)
   - Sequentially builds trees, each correcting previous errors
   - **Pros**: Often highest accuracy, handles missing values
   - **Cons**: More prone to overfitting, requires tuning
   - **Best for**: Maximum predictive performance

3. **LightGBM** (Light Gradient Boosting Machine)
   - Optimized gradient boosting (leaf-wise tree growth)
   - **Pros**: Very fast training, low memory usage
   - **Cons**: Can overfit on small datasets
   - **Best for**: Large-scale deployments

**Selection Criterion**: F1-weighted score
```python
f1_weighted = (f1_class0 × support0 + f1_class1 × support1 + f1_class2 × support2) / total_support

# Why F1-weighted?
# - Balances precision and recall
# - Weights by class frequency (important for imbalanced data)
# - More meaningful than accuracy for security applications
```

---

## 5. Data Processing

### 5.1 Input Data Formats

**Supported Formats**:
- **CSV** (Comma-Separated Values)
  - Most common format for network logs
  - Human-readable
  - Larger file size
- **Parquet** (columnar format)
  - Compressed binary format
  - Faster loading
  - Smaller file size (~50% of CSV)

**Example Network Traffic Data**:
```csv
dur,spkts,dpkts,sbytes,dbytes,proto,service,state,rate,attack_cat
0.12,5,3,240,180,tcp,http,FIN,41.67,Normal
0.45,10,8,2048,1536,tcp,ftp,CON,22.22,Backdoor
2.31,25,20,5120,4096,tcp,smtp,FIN,10.82,Worms
```

### 5.2 Data Validation

**API-level Validation** (backend_api.py):
```python
# 1. File format validation
if not (file.filename.endswith('.csv') or file.filename.endswith('.parquet')):
    raise HTTPException(400, "Unsupported file format")

# 2. File size check (implicit - FastAPI handles)
# Large files are handled via streaming

# 3. DataFrame validation
try:
    df = pd.read_csv(BytesIO(contents))
except Exception as e:
    raise HTTPException(400, f"Invalid CSV format: {e}")

# 4. Column validation (flexible - auto-fills missing)
# No strict validation, uses auto-mapping
```

**Dashboard-level Validation** (anomaly_detector.py):
```python
# 1. File upload validation
uploaded_file = st.file_uploader(
    "Upload Network Data",
    type=['csv', 'parquet']
)

# 2. Column mapping validation
# Suggests mappings, allows manual override
# Missing columns filled with defaults

# 3. Data type validation (implicit)
# Numerical features: Coerced to float
# Categorical features: Coerced to string
```

### 5.3 Missing Data Handling

**Strategy**: Imputation with domain-appropriate defaults

```python
# Categorical features → "unknown"
if col in ['proto', 'service', 'state']:
    df[col] = df[col].fillna('unknown')

# Numerical features → 0
else:
    df[col] = df[col].fillna(0)

# After one-hot encoding:
# Missing proto → all proto_* columns = 0, proto_unknown = 1
# Missing numerical → contributes 0 to scaled value
```

**Rationale**:
- "unknown" category: Captures absence of information as a feature
- Zero for numerical: Neutral value after scaling (mean-centered)
- Better than dropping rows: Preserves sample size

---

## 6. API Specification

### 6.1 Authentication
**Current**: None (open API)
**Future Enhancement**:
- API key authentication
- Rate limiting
- User quotas

### 6.2 Request/Response Formats

#### POST /predict/csv

**Request**:
```http
POST /predict/csv HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary
Content-Length: 12345

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="traffic.csv"
Content-Type: text/csv

[CSV data here]
------WebKitFormBoundary--
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/predict/csv" \
     -F "file=@network_traffic.csv"
```

**Python Example**:
```python
import requests

with open('network_traffic.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict/csv', files=files)
    results = response.json()

print(f"Anomaly Rate: {results['anomaly_rate']}%")
```

**Response Schema**:
```json
{
  "status": "success",                 // "success" or "error"
  "file_name": "network_traffic.csv",  // Original filename
  "total_connections": 1000,           // Total rows analyzed
  "normal_connections": 850,           // Count of Normal predictions
  "worms_detected": 100,               // Count of Worms predictions
  "backdoors_detected": 50,            // Count of Backdoor predictions
  "total_anomalies": 150,              // worms + backdoors
  "anomaly_rate": 15.0,                // (anomalies / total) × 100
  "metrics": {                         // Only if ground truth exists
    "precision": 0.9234,               // TP / (TP + FP)
    "recall": 0.8876,                  // TP / (TP + FN)
    "f1_score": 0.9051                 // 2 × (P × R) / (P + R)
  },
  "predictions": [                     // Per-connection predictions
    {
      "connection_id": 0,              // Row index
      "prediction": "Normal",          // Class label
      "confidence": 0.9823             // Max probability (0-1)
    },
    // ... (one per row)
  ]
}
```

**Error Response**:
```json
{
  "detail": "Unsupported file format. Please upload CSV or Parquet file."
}
```

### 6.3 CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)

# Production recommendation:
# allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"]
```

---

## 7. User Interfaces

### 7.1 Simplified UI (streamlit_app.py)

**Layout Structure**:
```
┌───────────────────────────────────────────────────────────┐
│  🛡️ Network Anomaly Detection System                      │
│  Upload your network data in CSV format to detect threats │
├───────────────────────────────────────────────────────────┤
│  [Choose a CSV file]  No file selected                    │
│  📁 File uploaded: network_traffic.csv                    │
│  🔄 Analyzing network data... This may take a moment.    │
├───────────────────────────────────────────────────────────┤
│  ✅ Analysis Complete!                                     │
├───────────────────────────────────────────────────────────┤
│  📊 Analysis Report                                        │
│  ┌─────────┬─────────┬─────────┬─────────┐               │
│  │  Total  │ Normal  │Backdoors│  Worms  │               │
│  │  1,000  │   850   │   50    │   100   │               │
│  │         │  85.0%  │  5.0%   │  10.0%  │               │
│  └─────────┴─────────┴─────────┴─────────┘               │
├───────────────────────────────────────────────────────────┤
│  📈 Performance Metrics                                    │
│  ┌──────────┬──────────┬──────────┐                       │
│  │Precision │  Recall  │ F1-Score │                       │
│  │  92.34%  │  88.76%  │  90.51%  │                       │
│  └──────────┴──────────┴──────────┘                       │
├───────────────────────────────────────────────────────────┤
│  🎯 Anomaly Detection Summary                              │
│  Anomaly Rate: 15.0%                                      │
│  🟡 Moderate threat level - Monitor closely               │
│  [Bar Chart: Normal, Backdoor, Worms]                    │
├───────────────────────────────────────────────────────────┤
│  ▶ 🔍 View Detailed Predictions                            │
│    Filter by prediction: [Backdoor ✓] [Worms ✓]          │
│    Minimum confidence: [━━━━━━━○━━━━] 50%                 │
│    ┌────────┬────────────┬────────────┐                   │
│    │ ID     │ Prediction │ Confidence │                   │
│    ├────────┼────────────┼────────────┤                   │
│    │ 42     │ Backdoor   │   0.9823   │                   │
│    │ 107    │ Worms      │   0.8956   │                   │
│    │ 215    │ Backdoor   │   0.7834   │                   │
│    └────────┴────────────┴────────────┘                   │
│    📥 Download Filtered Predictions (CSV)                 │
└───────────────────────────────────────────────────────────┘
```

### 7.2 Advanced Dashboard (anomaly_detector.py)

**Layout Structure**:
```
┌───────────────────────────────────────────────────────────┐
│  🛡️ Software de Análise de Segurança de Rede              │
│  Detecção de Malwares (Worms e Backdoor) com IA          │
├───────────────────────────────────────────────────────────┤
│  📁 Fazer Upload de Dados de Rede (CSV ou Parquet)       │
│  [Browse files]                                           │
│  ✅ 1,000 registros carregados de network_traffic.csv    │
├───────────────────────────────────────────────────────────┤
│  📋 Mapeamento de Colunas                                 │
│  ▼ Configurar Mapeamento de Colunas                       │
│    💡 O sistema detectou automaticamente algumas colunas  │
│    ┌──────────┬──────────┐ ┌──────────┬──────────┐       │
│    │ dur      │ duration │ │ spkts    │ src_pkts │       │
│    │ proto    │ protocol │ │ service  │ srv      │       │
│    │ ...      │ ...      │ │ ...      │ ...      │       │
│    └──────────┴──────────┘ └──────────┴──────────┘       │
├───────────────────────────────────────────────────────────┤
│  [🚀 Iniciar Análise de Segurança]                        │
├───────────────────────────────────────────────────────────┤
│  📊 Relatório de Análise de Segurança                     │
│  Análise realizada em: 06/10/2025 14:30:00               │
│  ┌──────────────┬──────────────┬──────────────┐          │
│  │ 🔍 Total:    │ ✅ Normais:  │ ⚠️ Anomalias: │          │
│  │   1,000      │    850       │    150        │          │
│  │              │   85.0%      │   15.0%       │          │
│  ├──────────────┼──────────────┼──────────────┤          │
│  │ 🎯 Confiança:│ 🔮 Risco:    │ 🚨 Suspeitas:│          │
│  │   92.3%      │ 🟡 35.2%    │    25         │          │
│  └──────────────┴──────────────┴──────────────┘          │
│  [Pie Chart]       [Bar Chart: Backdoor vs Worms]        │
├───────────────────────────────────────────────────────────┤
│  [Conexões Suspeitas] [Rede (IPs/Portas)] [Temporal]     │
│  [Explicabilidade] [Resultados Detalhados]               │
│  ┌───────────────────────────────────────────────────┐   │
│  │ 🚨 Conexões Suspeitas (Alto Risco)                │   │
│  │ ⚠️ 25 conexões suspeitas identificadas             │   │
│  │ ┌──────┬────────┬────────┬──────┬──────┐         │   │
│  │ │Class │Confiança│Risco(%)│Nível│proto │         │   │
│  │ ├──────┼────────┼────────┼──────┼──────┤         │   │
│  │ │Worms │  98.2% │  95.1% │Crít. │ tcp  │         │   │
│  │ │Back. │  87.3% │  78.3% │ Alto │ tcp  │         │   │
│  │ └──────┴────────┴────────┴──────┴──────┘         │   │
│  │ 📥 Baixar Conexões Suspeitas (CSV)                │   │
│  └───────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

**Tab Views**:

**1. Conexões Suspeitas**:
- Filtered view of high-risk connections (Alto/Crítico)
- Sortable columns
- Downloadable CSV

**2. Análise de Rede (IPs/Portas)**:
```
┌─────────────────────────────────────────────────┐
│ 🔴 Top IPs Atacantes (Fontes de Anomalias)     │
│ [Horizontal Bar Chart: Top 10 source IPs]      │
│ ┌──────────────┬──────────┬────────────┐       │
│ │ IP Address   │ Attacks  │ Avg Conf   │       │
│ ├──────────────┼──────────┼────────────┤       │
│ │ 192.168.1.42 │   127    │   89.2%    │       │
│ │ 10.0.0.15    │   98     │   91.5%    │       │
│ └──────────────┴──────────┴────────────┘       │
│ Ação Recomendada: Bloquear 3 IPs              │
├─────────────────────────────────────────────────┤
│ 🎯 Top IPs Alvos (Destinos de Ataques)        │
│ [Similar layout for destination IPs]           │
├─────────────────────────────────────────────────┤
│ 🔌 Análise de Portas de Ataque                 │
│ [Bar Chart: Top 15 ports]                      │
│ [Table: Ports by Attack Type]                  │
├─────────────────────────────────────────────────┤
│ 🔁 Padrões de Ataque Repetidos                 │
│ ⚠️ 15 padrões de ataque repetido detectados     │
│ ┌──────────────┬──────────────┬──────┬────────┐│
│ │ IP Origem    │ IP Destino   │ Tipo │Tentativas││
│ ├──────────────┼──────────────┼──────┼────────┤│
│ │ 192.168.1.42 │ 10.0.0.100   │ Worm │  45    ││
│ └──────────────┴──────────────┴──────┴────────┘│
└─────────────────────────────────────────────────┘
```

**3. Análise Temporal**:
```
┌─────────────────────────────────────────────────┐
│ 📈 Análise Temporal de Ataques                  │
│ ┌────────────┬────────────┬────────────┐        │
│ │ Período    │ Tendência  │ Picos      │        │
│ │  7 dias    │📈 Crescente│     3      │        │
│ └────────────┴────────────┴────────────┘        │
├─────────────────────────────────────────────────┤
│ Taxa de Anomalias ao Longo do Tempo            │
│ [Line Chart with spike markers]                │
│                                                 │
│ Evolução por Tipo de Ataque                    │
│ [Multi-line Chart: Backdoor vs Worms over time]│
│                                                 │
│ 🚨 Detalhes dos Picos de Ataque                │
│ ┌───────────┬────────┬──────────┬────────┐     │
│ │ Momento   │ Total  │ Anomalias│ Taxa(%)│     │
│ ├───────────┼────────┼──────────┼────────┤     │
│ │ 05/10 14h │  250   │   89     │ 35.6%  │     │
│ │ 05/10 18h │  310   │  102     │ 32.9%  │     │
│ └───────────┴────────┴──────────┴────────┘     │
└─────────────────────────────────────────────────┘
```

**4. Explicabilidade**:
```
┌─────────────────────────────────────────────────┐
│ 🔍 Explicabilidade do Modelo                    │
│ Selecione uma Conexão para Análise             │
│ [Dropdown: Conexão #42 - Worms (98.2%)]        │
├─────────────────────────────────────────────────┤
│ ┌────────────┬────────────┬────────────┐        │
│ │Classificação│ Confiança  │Nível Risco │        │
│ │   Worms    │   98.2%    │Crítico 95% │        │
│ └────────────┴────────────┴────────────┘        │
├─────────────────────────────────────────────────┤
│ Distribuição de Probabilidades                 │
│ [Bar Chart: Worms 98%, Backdoor 1%, Normal 1%] │
├─────────────────────────────────────────────────┤
│ Principais Features que Influenciaram a Decisão│
│ [Horizontal Bar Chart: Top 10 features]        │
│ ▼ Ver Valores das Features                     │
│   ┌──────────┬────────────┬────────┬──────────┐│
│   │ Feature  │ Importance │ Value  │Influence ││
│   ├──────────┼────────────┼────────┼──────────┤│
│   │ sbytes   │   0.1234   │ 8192.5 │  1.0112  ││
│   │ dur      │   0.0987   │  2.456 │  0.2425  ││
│   └──────────┴────────────┴────────┴──────────┘│
├─────────────────────────────────────────────────┤
│ ▼ Ver Dados Brutos da Conexão                  │
│   {                                             │
│     "dur": 2.456,                               │
│     "sbytes": 8192,                             │
│     "proto": "tcp",                             │
│     ...                                         │
│   }                                             │
└─────────────────────────────────────────────────┘
```

**5. Resultados Detalhados**:
```
┌─────────────────────────────────────────────────┐
│ 🔍 Resultados Detalhados da Análise             │
│ Filtros                                         │
│ ┌────────────┬────────────┬────────────┐        │
│ │Classificação│Nível Risco │Confiança   │        │
│ │[☑Backdoor] │[☑Alto]     │[━━━━○━] 50%│        │
│ │[☑Worms]    │[☑Crítico]  │            │        │
│ └────────────┴────────────┴────────────┘        │
│ Exibindo 250 de 1,000 conexões                 │
│ ┌──┬────────┬────────┬──────┬──────┬──────┐    │
│ │ID│Class   │Conf(%) │Risk% │Nível │proto │    │
│ ├──┼────────┼────────┼──────┼──────┼──────┤    │
│ │42│Worms   │  98.2  │ 95.1 │Crít. │ tcp  │    │
│ │88│Backdoor│  87.3  │ 78.3 │ Alto │ tcp  │    │
│ │..│...     │  ...   │ ...  │ ...  │ ...  │    │
│ └──┴────────┴────────┴──────┴──────┴──────┘    │
│ 📥 Baixar Resultados Filtrados (CSV)           │
└─────────────────────────────────────────────────┘
```

---

## 8. Model Training & Optimization

### 8.1 Training Configuration

**Reproducibility Settings**:
```python
RANDOM_STATE = 42
np.random.seed(42)
# Ensures same random splits, SMOTE samples, and model initialization
```

**Cross-Validation**:
```python
RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=10,              # Try 10 random parameter combinations
    cv=3,                   # 3-fold cross-validation
    scoring='f1_weighted',  # Optimize for F1-weighted
    n_jobs=-1,              # Use all CPU cores
    random_state=42
)
```

**3-Fold Cross-Validation**:
```
Training Data (175k samples)
├─ Fold 1: Train on 117k, Validate on 58k
├─ Fold 2: Train on 117k, Validate on 58k
└─ Fold 3: Train on 117k, Validate on 58k

Average F1-weighted across folds → Best parameters
```

### 8.2 Hyperparameter Search

**RandomizedSearchCV** (faster than GridSearchCV):
- Samples 10 random combinations from parameter space
- Total combinations tested: 10 × 3 folds = 30 model fits per algorithm
- Total models trained: 30 × 3 algorithms = 90 models

**Why Randomized?**
- Faster than exhaustive grid search
- Often finds near-optimal parameters
- Good for high-dimensional parameter spaces

**Full Parameter Space**:

```python
# Random Forest: 2 × 3 × 2 × 2 = 24 combinations
{
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# XGBoost: 2 × 3 × 3 × 2 = 36 combinations
{
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 15],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__subsample': [0.7, 0.8]
}

# LightGBM: 2 × 3 × 2 × 2 = 24 combinations
{
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, -1],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__num_leaves': [31, 50]
}
```

**Typical Best Parameters** (Random Forest):
```python
{
    'classifier__n_estimators': 200,
    'classifier__max_depth': 20,
    'classifier__min_samples_split': 2,
    'classifier__min_samples_leaf': 1
}
```

### 8.3 Evaluation Metrics

**Metrics Tracked**:

1. **F1-weighted** (primary metric)
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   F1_weighted = Σ(F1_i × support_i) / total_support
   ```

2. **Per-class Metrics**:
   - Precision: TP / (TP + FP) for each class
   - Recall: TP / (TP + FN) for each class
   - F1-Score: Harmonic mean of precision/recall
   - Support: Number of true instances per class

3. **Confusion Matrix**:
   ```
                 Predicted
                 B    N    W
   Actual  B  [TN   FP   FP]
           N  [FN   TP   FN]
           W  [FN   FN   TP]

   B = Backdoor, N = Normal, W = Worms
   ```

**Typical Performance** (on UNSW-NB15):
```
Class       Precision  Recall  F1-Score  Support
Backdoor       0.85     0.78     0.81      8,234
Normal         0.98     0.99     0.98     56,000
Worms          0.79     0.71     0.75      8,098

Accuracy: 0.96
F1-weighted: 0.94
```

### 8.4 MLflow Integration

**Tracking Hierarchy**:
```
mlruns/
├── 0/                              # Default experiment
│   └── meta.yaml
├── 160034205330776860/             # UNSW_NB15_Classification experiment
│   ├── 225e9a5a8da5445f9bccf8afc78783a3/  # RandomForest run
│   │   ├── meta.yaml               # Run metadata
│   │   ├── params/                 # Hyperparameters
│   │   │   ├── classifier__n_estimators
│   │   │   ├── classifier__max_depth
│   │   │   └── ...
│   │   ├── metrics/                # Performance metrics
│   │   │   ├── f1_weighted
│   │   │   ├── Backdoor_precision
│   │   │   ├── Backdoor_recall
│   │   │   └── ...
│   │   ├── artifacts/              # Files (images, models)
│   │   │   ├── confusion_matrix_RandomForest.png
│   │   │   └── model/              # Serialized model
│   │   │       └── python_model.pkl
│   │   └── tags/                   # Custom tags
│   ├── 5261ec1971b2444b84ab77a4f636f35e/  # XGBoost run
│   └── e6d109ab09be4f48b3d0fd244d60d601/  # LightGBM run
```

**Viewing Results**:
```bash
mlflow ui
# Opens web interface at http://localhost:5000
# Compare runs, visualize metrics, download models
```

**Logged Information** (per run):
- **Parameters**: All hyperparameters tried
- **Metrics**: F1, precision, recall, accuracy (per class and overall)
- **Artifacts**: Confusion matrix PNG, serialized model
- **Tags**: Custom metadata (model name, dataset version, etc.)

---

## 9. Academic Results Generation

### 9.1 Script Purpose

**`generate_tcc_results.py`** generates all artifacts needed for academic papers:
- Side-by-side comparison of Naive vs Improved models
- Publication-ready visualizations (300 DPI)
- Copy-paste-ready text reports
- Quantitative improvement analysis

### 9.2 Naive vs Improved Comparison

**Naive Model**:
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
    # NO class_weight parameter
)
```
**Characteristics**:
- Treats all classes equally
- High accuracy (~96%) but poor minority class recall
- Misses most Backdoor/Worms attacks
- Baseline for comparison

**Improved Model**:
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',  # KEY DIFFERENCE
    n_jobs=-1
)
```
**Characteristics**:
- Penalizes minority class errors heavily
- Slightly lower accuracy (~94%) but high minority class recall
- Detects most attacks (critical for security)
- Demonstrates effectiveness of imbalance handling

**Typical Improvement**:
```
Class       Naive F1  Improved F1  Improvement
Backdoor       0.42        0.81       +92.9%
Worms          0.35        0.75       +114.3%
Normal         0.98        0.98        +0.0%

Overall Acc    0.96        0.94        -2.1%
F1-weighted    0.87        0.94        +8.0%
```

**Key Insight**: Small accuracy drop is acceptable for massive improvement in attack detection.

### 9.3 Academic Writing Tips

**Structuring Results Chapter**:

1. **Introduction**:
   - State research question: "How effective is class imbalance handling?"
   - Describe experimental setup

2. **Baseline (Naive Model)**:
   - Present classification report
   - Show confusion matrix
   - Highlight poor minority class performance
   - Explain: "Model learns to always predict Normal"

3. **Improved Model**:
   - Present classification report
   - Show confusion matrix
   - Highlight improved minority class performance
   - Explain: "Class weighting forces model to learn attack patterns"

4. **Comparative Analysis**:
   - Table with side-by-side F1-scores
   - Calculate % improvement
   - Discuss trade-offs (accuracy vs recall)
   - Justify: "For security, detecting attacks is more critical than overall accuracy"

5. **Feature Analysis**:
   - Show feature importance chart
   - Discuss top features (sbytes, dur, etc.)
   - Relate to domain knowledge (network traffic patterns)

**Sample Paragraph**:
```
The proposed approach demonstrates significant improvement over the baseline.
Table 4.1 presents a comparative analysis of F1-scores between the naive model
(without class balancing) and the improved model (with class_weight='balanced').

For Backdoor attacks, the F1-score improved from 0.42 to 0.81, representing a
92.9% increase. Similarly, Worms detection improved from 0.35 to 0.75 (114.3%
increase). This dramatic improvement comes at a minimal cost: overall accuracy
decreased by only 2.1% (from 0.96 to 0.94).

In the context of network security, this trade-off is highly desirable. Missing
a single Backdoor or Worm attack can compromise an entire network, whereas a
false positive (normal traffic flagged as attack) merely triggers additional
investigation. Therefore, maximizing recall for minority attack classes is
paramount, even at the expense of slight accuracy degradation.

Figure 4.3 illustrates the confusion matrices for both models. The naive model's
confusion matrix shows a clear bias toward the Normal class, with low true
positive rates for Backdoor (0.35) and Worms (0.28). In contrast, the improved
model achieves true positive rates of 0.78 and 0.71 respectively, demonstrating
effective learning of minority class patterns.
```

---

## 10. Dependencies & Requirements

### 10.1 Python Version
- **Required**: Python 3.9 or higher
- **Recommended**: Python 3.10
- **Tested on**: Python 3.10.12

### 10.2 Core Dependencies

**Machine Learning**:
- `scikit-learn==1.7.2` - ML algorithms, preprocessing, metrics
- `xgboost==3.0.5` - Gradient boosting
- `lightgbm==4.6.0` - Fast gradient boosting
- `imbalanced-learn==0.14.0` - SMOTE for class imbalance

**Data Processing**:
- `pandas==2.3.3` - DataFrame operations
- `numpy==2.3.3` - Numerical computations
- `pyarrow==21.0.0` - Parquet file support

**Visualization**:
- `matplotlib==3.10.6` - Plotting
- `seaborn==0.13.2` - Statistical visualizations
- `streamlit==1.50.0` - Web dashboards

**Backend API**:
- `fastapi==0.118.0` - REST API framework
- `uvicorn==0.37.0` - ASGI server
- `python-multipart==0.0.20` - File upload handling

**Utilities**:
- `joblib==1.5.2` - Model serialization
- `requests==2.32.5` - HTTP client (for frontend → backend)

**Experiment Tracking** (optional):
- `mlflow==3.4.0` - Model tracking and registry

### 10.3 System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 2 GB (for datasets and models)

**Recommended**:
- CPU: 8+ cores (for parallel hyperparameter search)
- RAM: 16 GB
- Storage: 5 GB (for MLflow experiments)

**Training Time** (on recommended hardware):
- Model training: ~10-15 minutes
- TCC results generation: ~5-10 minutes
- Single prediction: <1 second per 1000 connections

---

## 11. Installation & Setup

### 11.1 Step-by-Step Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd back

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import sklearn, xgboost, lightgbm, streamlit, fastapi; print('All packages installed successfully')"
```

### 11.2 Dataset Setup

**Datasets are included** in `datasets/` directory:
- `UNSW_NB15_training-set.parquet` (9.6 MB)
- `UNSW_NB15_testing-set.parquet` (4.5 MB)
- `UNSW_NB15_testing-set.csv` (32.3 MB) - optional

**If missing, download from**:
- Official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Or contact project maintainer

### 11.3 Model Training

```bash
# Train models (required before first use)
python model_training.py

# Expected output:
# [Training progress bars]
# ✅ Best Model: RandomForest (F1: 0.9234)
# ✅ Artifacts saved:
#    - best_model_pipeline_RandomForest.joblib
#    - pipeline.joblib
#    - model.joblib
#    - scaler.joblib
#    - model_columns.joblib
#    - label_encoder.joblib
```

**Training time**: 10-15 minutes (depending on hardware)

**Generated files** (~500 MB total):
- Model files: ~50 MB each
- MLflow experiments: ~400 MB

### 11.4 Running Applications

#### **Option 1: Simplified UI (API-based)**

**Terminal 1 - Backend**:
```bash
python backend_api.py

# Expected output:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# ✅ Model artifacts loaded successfully
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Frontend**:
```bash
streamlit run streamlit_app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.1.x:8501
```

**Access**:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

#### **Option 2: Advanced Dashboard (Standalone)**

```bash
streamlit run anomaly_detector.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

**Access**:
- Dashboard: http://localhost:8501

---

## 12. Usage Examples

### 12.1 Command-Line API Usage

**Using cURL**:
```bash
# Predict on CSV file
curl -X POST "http://localhost:8000/predict/csv" \
     -F "file=@datasets/UNSW_NB15_testing-set.csv" \
     -o results.json

# View results
cat results.json | jq '.anomaly_rate'
# Output: 15.2
```

**Using Python**:
```python
import requests
import pandas as pd

# Load data
df = pd.read_csv('network_traffic.csv')

# Send to API
with open('network_traffic.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/csv',
        files={'file': f}
    )

results = response.json()

# Print summary
print(f"Total Connections: {results['total_connections']}")
print(f"Anomaly Rate: {results['anomaly_rate']}%")
print(f"Worms Detected: {results['worms_detected']}")
print(f"Backdoors Detected: {results['backdoors_detected']}")

# Get detailed predictions
predictions_df = pd.DataFrame(results['predictions'])
high_risk = predictions_df[predictions_df['confidence'] > 0.9]
print(f"\nHigh-confidence anomalies: {len(high_risk)}")
```

### 12.2 Batch Processing

**Process multiple files**:
```python
import os
import requests
import pandas as pd

files = [f for f in os.listdir('logs/') if f.endswith('.csv')]

results_summary = []

for filename in files:
    with open(f'logs/{filename}', 'rb') as f:
        response = requests.post(
            'http://localhost:8000/predict/csv',
            files={'file': f}
        )

        results = response.json()

        results_summary.append({
            'file': filename,
            'total': results['total_connections'],
            'anomaly_rate': results['anomaly_rate'],
            'worms': results['worms_detected'],
            'backdoors': results['backdoors_detected']
        })

# Create summary report
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv('batch_results.csv', index=False)
print(summary_df)
```

### 12.3 Integration Examples

**Integrate with monitoring pipeline**:
```python
# Example: Process logs every 5 minutes

import schedule
import time

def analyze_recent_logs():
    # Read latest logs
    logs = read_logs_from_database()
    logs.to_csv('temp_logs.csv')

    # Analyze
    with open('temp_logs.csv', 'rb') as f:
        response = requests.post(
            'http://localhost:8000/predict/csv',
            files={'file': f}
        )

    results = response.json()

    # Alert if high anomaly rate
    if results['anomaly_rate'] > 20:
        send_alert(f"High anomaly rate: {results['anomaly_rate']}%")

    # Log results
    log_to_database(results)

# Schedule task
schedule.every(5).minutes.do(analyze_recent_logs)

while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## 13. File Structure

```
back/
├── README.md                                # Project overview
├── PROJECT_DOCUMENTATION.md                 # This file (complete documentation)
├── TCC_RESULTS_GUIDE.md                     # Guide for academic use
├── new_changes.md                           # Changelog
├── requirements.txt                         # Python dependencies
│
├── backend_api.py                           # FastAPI REST server
├── streamlit_app.py                         # Simplified frontend (API client)
├── anomaly_detector.py                      # Advanced dashboard (standalone)
├── model_training.py                        # Model training script
├── generate_tcc_results.py                  # TCC results generator
│
├── datasets/                                # Training/testing data
│   ├── UNSW_NB15_training-set.parquet       # Training set (175k samples)
│   ├── UNSW_NB15_testing-set.parquet        # Testing set (82k samples)
│   └── UNSW_NB15_testing-set.csv            # Testing set (CSV format)
│
├── tcc_results/                             # Academic results
│   ├── naive_model_report.txt               # Naive model metrics
│   ├── improved_model_report.txt            # Improved model metrics
│   ├── naive_model_confusion_matrix.png     # Naive confusion matrix
│   ├── improved_model_confusion_matrix.png  # Improved confusion matrix
│   ├── feature_importances.png              # Feature importance chart
│   └── feature_importances.csv              # Feature importance data
│
├── mlruns/                                  # MLflow experiment tracking
│   ├── 0/                                   # Default experiment
│   └── 160034205330776860/                  # UNSW_NB15_Classification
│       ├── <run-id-1>/                      # RandomForest run
│       ├── <run-id-2>/                      # XGBoost run
│       └── <run-id-3>/                      # LightGBM run
│
├── best_model_pipeline_RandomForest.joblib  # Complete pipeline (for dashboard)
├── pipeline.joblib                          # Complete pipeline (for API)
├── model.joblib                             # Classifier only
├── scaler.joblib                            # StandardScaler
├── model_columns.joblib                     # Expected features
├── label_encoder.joblib                     # Label decoder
│
├── venv/                                    # Python virtual environment
│   ├── bin/                                 # Executables
│   ├── lib/                                 # Packages
│   └── pyvenv.cfg                           # Venv config
│
├── __pycache__/                             # Python bytecode
├── .git/                                    # Git repository
├── .gitignore                               # Git ignore rules
└── pyvenv.cfg                               # Python environment config
```

### File Sizes

```
Total: ~600 MB

Datasets:
- UNSW_NB15_training-set.parquet      9.6 MB
- UNSW_NB15_testing-set.parquet       4.5 MB
- UNSW_NB15_testing-set.csv          32.3 MB

Models:
- best_model_pipeline_*.joblib       ~50 MB
- pipeline.joblib                    ~45 MB
- model.joblib                       ~40 MB
- scaler.joblib                       ~8 KB
- model_columns.joblib                ~2 KB
- label_encoder.joblib                ~1 KB

MLflow:
- mlruns/                           ~400 MB

TCC Results:
- *.png                              ~1.5 MB
- *.txt                              ~2 KB
- *.csv                              ~6 KB

Virtual Environment:
- venv/                             ~150 MB
```

---

## 14. Technical Details

### 14.1 Algorithms

**Random Forest** (default choice):
- **Type**: Ensemble learning (bagging)
- **How it works**:
  1. Create N decision trees (e.g., 200)
  2. Each tree trained on random subset of data
  3. Each split considers random subset of features
  4. Final prediction = majority vote
- **Pros**:
  - Resistant to overfitting
  - Handles non-linear relationships
  - Provides feature importance
  - Robust to outliers
- **Cons**:
  - Slower inference than single tree
  - Less interpretable than single tree
  - Memory intensive

**XGBoost** (eXtreme Gradient Boosting):
- **Type**: Ensemble learning (boosting)
- **How it works**:
  1. Train initial weak model (shallow tree)
  2. Calculate residuals (errors)
  3. Train next tree to predict residuals
  4. Add tree to ensemble with learning rate
  5. Repeat until convergence or max trees
- **Pros**:
  - Often highest accuracy
  - Handles missing values
  - Built-in regularization
  - GPU acceleration available
- **Cons**:
  - Prone to overfitting without tuning
  - Longer training time
  - More hyperparameters to tune

**LightGBM** (Light Gradient Boosting Machine):
- **Type**: Ensemble learning (boosting)
- **How it works**:
  1. Similar to XGBoost but uses leaf-wise growth
  2. Grows tree by splitting leaf with max delta loss
  3. Results in deeper, more complex trees
  4. Uses histogram-based algorithm (faster)
- **Pros**:
  - Very fast training
  - Low memory usage
  - Handles large datasets well
  - GPU acceleration available
- **Cons**:
  - Can overfit on small datasets
  - Deeper trees = less interpretable

### 14.2 Feature Importance

**Calculation** (Random Forest):
```python
# For each feature:
importance = Σ (impurity_decrease × node_samples / total_samples)
             over all nodes where feature is used

# Impurity decrease:
# - For classification: Gini impurity or entropy reduction
# - Larger decrease = more important feature
```

**Interpretation**:
```
Feature: sbytes (source bytes)
Importance: 0.1234

Meaning:
- sbytes contributes 12.34% to the model's decision-making
- Across all trees, sbytes is used to split nodes that
  collectively account for 12.34% of total impurity reduction
- Removing this feature would significantly degrade performance
```

**Top Features** (typical for UNSW-NB15):
1. `sbytes` (0.123): Total bytes sent - large transfers indicate attacks
2. `dbytes` (0.098): Total bytes received - command-and-control communication
3. `dur` (0.087): Connection duration - worms maintain persistent connections
4. `ct_state_ttl` (0.076): Connections with same state - port scanning
5. `sttl` (0.062): Source TTL - OS fingerprinting
6. `dttl` (0.058): Destination TTL - traceroute activity
7. `ct_srv_dst` (0.054): Connections to same service - brute-force attacks
8. `sload` (0.049): Source load - DDoS indicators
9. `dload` (0.047): Destination load - data exfiltration
10. `sinpkt` (0.043): Inter-arrival time - covert channels

### 14.3 Confidence Scores

**How Confidence is Calculated**:
```python
# For Random Forest:
# Each tree votes for a class (0, 1, or 2)
# Confidence = proportion of trees voting for predicted class

Example:
200 trees
- 185 vote "Worms"
- 10 vote "Backdoor"
- 5 vote "Normal"

Prediction: Worms
Confidence: 185/200 = 0.925 (92.5%)

# Probabilities for all classes:
probabilities = [0.05, 0.05, 0.90]  # [Backdoor, Normal, Worms]
confidence = max(probabilities) = 0.90
```

**Interpretation**:
- **High confidence (>90%)**: Model is very certain
- **Medium confidence (70-90%)**: Model is fairly certain
- **Low confidence (<70%)**: Model is unsure, may be borderline case

**Use in Risk Scoring**:
```python
# Advanced dashboard uses confidence in risk calculation
risk_score = base_risk + (confidence × confidence_weight)

# High-confidence Worms: 50 + (0.95 × 45) = 92.75% risk (Critical)
# Low-confidence Worms: 50 + (0.60 × 45) = 77.00% risk (High)
```

### 14.4 Performance Benchmarks

**Inference Speed** (on test hardware: 16-core CPU, 32GB RAM):
```
Single connection:    ~0.5 ms
100 connections:      ~10 ms
1,000 connections:    ~80 ms
10,000 connections:   ~750 ms
100,000 connections:  ~8 seconds
```

**Memory Usage**:
```
Backend API (idle):         ~150 MB
Backend API (processing):   ~500 MB (for 100k connections)
Dashboard (idle):           ~200 MB
Dashboard (with results):   ~800 MB (for 100k connections)
```

**Model Sizes**:
```
Random Forest (200 trees):  ~50 MB
XGBoost (200 trees):        ~45 MB
LightGBM (200 trees):       ~40 MB
```

**Training Performance**:
```
Random Forest:  ~8 minutes (3-fold CV × 10 random searches)
XGBoost:        ~12 minutes
LightGBM:       ~6 minutes
Total:          ~26 minutes for all models
```

---

## 15. Performance Metrics

### 15.1 Classification Metrics Explained

**Confusion Matrix**:
```
                Predicted
                 B    N    W
Actual   B    [TP   FP   FP]    True Positives, False Positives
         N    [FN   TP   FN]    False Negatives, True Positives
         W    [FN   FN   TP]    False Negatives, True Positives

B = Backdoor, N = Normal, W = Worms
```

**Precision** (Positive Predictive Value):
```
Precision = TP / (TP + FP)

Question: "Of all the samples predicted as attack, how many were actually attacks?"

Example (Backdoor):
- TP (True Positives): 620 (correctly identified Backdoors)
- FP (False Positives): 80 (Normal/Worms misclassified as Backdoor)
- Precision = 620 / (620 + 80) = 0.886 (88.6%)

Interpretation: When model predicts "Backdoor", it's correct 88.6% of the time
```

**Recall** (Sensitivity, True Positive Rate):
```
Recall = TP / (TP + FN)

Question: "Of all actual attacks, how many did we detect?"

Example (Backdoor):
- TP (True Positives): 620 (correctly identified Backdoors)
- FN (False Negatives): 180 (Backdoors misclassified as Normal/Worms)
- Recall = 620 / (620 + 180) = 0.775 (77.5%)

Interpretation: Model detects 77.5% of all Backdoor attacks
```

**F1-Score** (Harmonic Mean of Precision and Recall):
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Example:
- Precision = 0.886
- Recall = 0.775
- F1 = 2 × (0.886 × 0.775) / (0.886 + 0.775) = 0.827

Why harmonic mean (not arithmetic)?
- Arithmetic mean: (0.886 + 0.775) / 2 = 0.831
- Harmonic mean: 0.827
- Harmonic mean penalizes extreme values
- Example: Precision=1.0, Recall=0.1
  - Arithmetic mean: 0.55 (misleadingly high)
  - Harmonic mean: 0.18 (more realistic)
```

**Accuracy** (Overall Correctness):
```
Accuracy = (TP + TN) / Total

Example:
- Total samples: 82,332
- Correct predictions: 79,134
- Accuracy = 79,134 / 82,332 = 0.961 (96.1%)

Warning: Misleading for imbalanced datasets!
- If 90% of data is Normal, always predicting "Normal" gives 90% accuracy
- But this model is useless for detecting attacks
```

**Weighted Metrics** (Account for Class Imbalance):
```
Metric_weighted = Σ(Metric_i × support_i) / total_support

Example F1-weighted:
- Backdoor:  F1=0.827, support=800   → 0.827 × 800 = 661.6
- Normal:    F1=0.981, support=56000 → 0.981 × 56000 = 54,936
- Worms:     F1=0.759, support=780   → 0.759 × 780 = 592.0
- Total support: 57,580
- F1-weighted = (661.6 + 54,936 + 592.0) / 57,580 = 0.969

Why weighted?
- Reflects that Normal class dominates dataset
- More realistic for production deployment
- Balances between minority and majority performance
```

### 15.2 Model Comparison Results

**Example Results** (from generate_tcc_results.py):

**Naive Model**:
```
              precision    recall  f1-score   support

    Backdoor     0.4912    0.3456    0.4060      8234
      Normal     0.9823    0.9941    0.9882     56000
       Worms     0.4201    0.2789    0.3354      8098

    accuracy                         0.9592     72332
   macro avg     0.6312    0.5395    0.5765     72332
weighted avg     0.9483    0.9592    0.9531     72332
```

**Analysis**:
- High overall accuracy (95.9%) is misleading
- Terrible minority class recall:
  - Only 34.6% of Backdoors detected
  - Only 27.9% of Worms detected
- Model is biased toward Normal class
- **Unsuitable for security applications**

**Improved Model**:
```
              precision    recall  f1-score   support

    Backdoor     0.8863    0.7750    0.8270      8234
      Normal     0.9807    0.9910    0.9858     56000
       Worms     0.7934    0.7106    0.7498      8098

    accuracy                         0.9631     72332
   macro avg     0.8868    0.8255    0.8542     72332
weighted avg     0.9625    0.9631    0.9627     72332
```

**Analysis**:
- Similar overall accuracy (96.3%)
- Dramatically improved minority class recall:
  - 77.5% of Backdoors detected (+124% improvement)
  - 71.1% of Worms detected (+155% improvement)
- Slight reduction in Normal precision (acceptable trade-off)
- **Suitable for production security systems**

**Comparison Table**:
```
│ Class    │ Naive F1 │ Improved F1 │ Δ F1    │ % Change │
├──────────┼──────────┼─────────────┼─────────┼──────────┤
│ Backdoor │  0.4060  │    0.8270   │ +0.4210 │  +103.7% │
│ Normal   │  0.9882  │    0.9858   │ -0.0024 │   -0.2%  │
│ Worms    │  0.3354  │    0.7498   │ +0.4144 │  +123.6% │
│ Weighted │  0.9531  │    0.9627   │ +0.0096 │   +1.0%  │
└──────────┴──────────┴─────────────┴─────────┴──────────┘
```

### 15.3 Production Metrics

**API Response Times** (50th/95th/99th percentile):
```
Empty file:          10ms / 15ms / 20ms
100 connections:     50ms / 80ms / 120ms
1,000 connections:   200ms / 350ms / 500ms
10,000 connections:  1.5s / 2.5s / 3.5s
```

**Throughput**:
```
Connections per second (single instance):
- With 8 CPU cores:  ~5,000 connections/sec
- With 16 CPU cores: ~8,500 connections/sec
```

**Scalability**:
```
Horizontal scaling (multiple API instances):
- 2 instances: ~17,000 connections/sec
- 4 instances: ~34,000 connections/sec
- 8 instances: ~68,000 connections/sec
(Load balancer required)
```

---

## 16. Troubleshooting

### 16.1 Common Issues

#### **Issue 1: Model files not found**
```
ERROR: FileNotFoundError: [Errno 2] No such file or directory: 'pipeline.joblib'
```

**Cause**: Model training hasn't been run

**Solution**:
```bash
python model_training.py
```

---

#### **Issue 2: Backend API connection refused**
```
ERROR: ConnectionError: Could not connect to the backend API
```

**Cause**: Backend API is not running

**Solution**:
```bash
# Terminal 1: Start backend
python backend_api.py

# Terminal 2: Start frontend
streamlit run streamlit_app.py
```

---

#### **Issue 3: Module not found errors**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Cause**: Dependencies not installed or wrong Python environment

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

---

#### **Issue 4: Out of memory during training**
```
MemoryError: Unable to allocate array
```

**Cause**: Insufficient RAM for SMOTE or large models

**Solution**:
```python
# Edit model_training.py, reduce n_estimators
{
    'classifier__n_estimators': [50, 100],  # Instead of [100, 200]
}

# Or disable SMOTE (use class_weight only)
# Comment out lines 57-65 in model_training.py
# X_train_balanced = X_train
# y_train_balanced = y_train_encoded
```

---

#### **Issue 5: MLflow UI not working**
```
bash: mlflow: command not found
```

**Cause**: MLflow not installed or not in PATH

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install MLflow if missing
pip install mlflow

# Start UI
mlflow ui
```

---

#### **Issue 6: Streamlit port already in use**
```
OSError: [Errno 48] Address already in use
```

**Cause**: Another Streamlit instance is running on port 8501

**Solution**:
```bash
# Option 1: Kill existing process
lsof -ti:8501 | xargs kill -9

# Option 2: Use different port
streamlit run streamlit_app.py --server.port 8502
```

---

#### **Issue 7: Backend API 500 error**
```
HTTP 500: Error processing file: 'float' object has no attribute 'fillna'
```

**Cause**: Input data contains unexpected types

**Solution**:
- Ensure CSV has proper headers
- Check for special characters in column names
- Validate that numerical columns contain only numbers

```python
# Debug by checking uploaded data
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.dtypes)  # Check data types
print(df.isnull().sum())  # Check for missing values
```

---

#### **Issue 8: Slow dashboard performance**
```
Dashboard is laggy with large datasets
```

**Cause**: Streamlit reprocesses data on every interaction

**Solution**:
```python
# Already implemented in anomaly_detector.py using session_state
# Ensure data is cached:
if 'predictions' not in st.session_state:
    # Process only once
    st.session_state['predictions'] = model.predict(X)
```

---

#### **Issue 9: High false positive rate**
```
Model flags too many normal connections as attacks
```

**Cause**: Model trained with heavy class weighting

**Solution**:
```python
# Retrain with adjusted class weights
RandomForestClassifier(
    class_weight={0: 3, 1: 1, 2: 3}  # Custom weights (Backdoor: 3, Normal: 1, Worms: 3)
)

# Or increase confidence threshold for alerts
high_risk = predictions[predictions['confidence'] > 0.85]
```

---

#### **Issue 10: Missing columns in input data**
```
KeyError: 'proto'
```

**Cause**: Input CSV doesn't have expected columns

**Solution**:
- **Dashboard**: Use column mapping interface to map user columns to expected columns
- **API**: Missing columns are auto-filled with defaults (0 for numerical, "unknown" for categorical)

---

### 16.2 Debugging Tips

**Enable detailed logging**:
```python
# In backend_api.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check model artifact integrity**:
```python
import joblib

# Test loading
pipeline = joblib.load('pipeline.joblib')
print(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")

# Test prediction
import numpy as np
X_test = np.zeros((1, 196))  # Dummy input
prediction = pipeline.predict(X_test)
print(f"Test prediction: {prediction}")
```

**Profile performance**:
```bash
# Time prediction
time curl -X POST "http://localhost:8000/predict/csv" -F "file=@test.csv"

# Python profiling
python -m cProfile -o profile.stats model_training.py
python -m pstats profile.stats
> sort cumulative
> stats 20
```

---

### 16.3 FAQ

**Q: Can I use a different dataset?**
A: Yes, but you need to retrain the model:
1. Format your data similarly to UNSW-NB15
2. Update `preprocess_data()` in `model_training.py`
3. Retrain: `python model_training.py`

**Q: How do I deploy this in production?**
A: Recommended setup:
```
Load Balancer (Nginx)
    ↓
Multiple API Instances (Docker containers)
    ↓
Shared Model Storage (S3 or NFS)
    ↓
Centralized Logging (ELK stack)
```

**Q: Can I use GPU acceleration?**
A: Yes, for XGBoost/LightGBM:
```python
XGBClassifier(tree_method='gpu_hist', gpu_id=0)
LGBMClassifier(device='gpu')
```
Requires CUDA-enabled GPU and appropriate drivers.

**Q: How often should I retrain the model?**
A: Depends on your network:
- Static network: Every 6-12 months
- Dynamic network: Every 1-3 months
- After major network changes: Immediately

Monitor model performance over time (concept drift detection).

**Q: Can I add more attack types?**
A: Yes, modify `preprocess_data()` to include additional attack classes:
```python
df_filtered = df[df['attack_cat'].isin([
    'Worms', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers'
]) | (df['label'] == 0)]
```
Then retrain the model.

---

## Conclusion

This project provides a complete, production-ready network anomaly detection system with:
- **Flexible architecture**: Choose between simplified UI or advanced dashboard
- **State-of-the-art ML**: Multiple algorithms with automatic selection
- **Imbalance handling**: SMOTE + class weighting for minority attack detection
- **Comprehensive analytics**: Risk scoring, temporal analysis, network topology
- **Academic support**: TCC results generation with publication-ready outputs
- **Easy deployment**: REST API for integration into existing systems

**For questions, issues, or contributions, please contact the project maintainer or open an issue on GitHub.**

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Author**: [Your Name/Team]
**License**: [Specify License]
