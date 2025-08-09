import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump

# Load Datasets
df_train = pd.read_parquet("./datasets/UNSW_NB15_training-set.parquet")
df_test = pd.read_parquet("./datasets/UNSW_NB15_testing-set.parquet")


def preprocess_data(df):
    """Preprocesses the raw UNSW-NB15 data."""
    # We keep the original filtering for now, but consider expanding it
    df_filtered = df[df['attack_cat'].isin(['Worms', 'Backdoor']) | (df['label'] == 0)].copy()
    df_filtered['attack_label'] = df_filtered['attack_cat'].fillna('Normal')
    cols_to_drop = [col for col in ['id', 'label', 'attack_cat'] if col in df_filtered.columns]
    df_filtered = df_filtered.drop(columns=cols_to_drop)
    X = df_filtered.drop(columns=['attack_label'])
    y = df_filtered['attack_label']
    return X, y

X_train_raw, y_train = preprocess_data(df_train)
X_test_raw, y_test = preprocess_data(df_test)

# --- NEW: Display class distribution to check for imbalance ---
print("--- Class Distribution in Training Set ---")
print(y_train.value_counts())
print("-" * 40)

# One-hot encode and align columns
X_train = pd.get_dummies(X_train_raw)
X_test = pd.get_dummies(X_test_raw)

train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[X_train.columns]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- IMPROVEMENT 1: Hyperparameter Tuning with RandomizedSearchCV ---
# Define the parameter grid to search
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# --- IMPROVEMENT 2: Add class_weight='balanced' to handle imbalance ---
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Set up RandomizedSearchCV
# n_iter trades off runtime vs. search quality. 10-20 is often a good start.
# n_jobs=-1 uses all available CPU cores.
rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter search with RandomizedSearchCV...")
rf_random_search.fit(X_train_scaled, y_train)
print("Search complete.")

# Get the best model from the search
best_model = rf_random_search.best_estimator_
print("\nBest Model Parameters:", rf_random_search.best_params_)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_scaled)

print("\n--- Classification Report for Best Model ---")
class_labels = sorted(pd.concat([y_train, y_test]).unique())
print(classification_report(y_test, y_pred, target_names=class_labels))

print("\n--- Confusion Matrix for Best Model ---")
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Best Model')
plt.show()

print("\n--- Saving Best Model and Artifacts ---")
dump(best_model, 'model.joblib')
dump(scaler, 'scaler.joblib')
dump(X_train.columns, 'model_columns.joblib')
print("Model, scaler, and columns saved successfully.")