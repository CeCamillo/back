import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # We will use this now
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump

df_train = pd.read_parquet("./datasets/UNSW_NB15_training-set.parquet")
df_test = pd.read_parquet("./datasets/UNSW_NB15_testing-set.parquet")


def preprocess_data(df):
    """Preprocesses the raw UNSW-NB15 data."""
    df_filtered = df[df['attack_cat'].isin(['Worms', 'Backdoor']) | (df['label'] == 0)].copy()

    df_filtered['attack_label'] = df_filtered['attack_cat'].fillna('Normal')

    cols_to_drop = [col for col in ['id', 'label', 'attack_cat'] if col in df_filtered.columns]
    df_filtered = df_filtered.drop(columns=cols_to_drop)

    X = df_filtered.drop(columns=['attack_label'])
    y = df_filtered['attack_label']

    return X, y

X_train_raw, y_train = preprocess_data(df_train)
X_test_raw, y_test = preprocess_data(df_test)


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



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Training the RandomForestClassifier...")
model.fit(X_train_scaled, y_train)
print("Training complete.")

y_pred = model.predict(X_test_scaled)

print("\n--- Classification Report ---")
class_labels = sorted(pd.concat([y_train, y_test]).unique())
print(classification_report(y_test, y_pred, target_names=class_labels))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\n--- Saving Model and Artifacts ---")
dump(model, 'model.joblib')
dump(scaler, 'scaler.joblib')
dump(X_train.columns, 'model_columns.joblib')
print("Model, scaler, and columns saved successfully.")