import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD DATA
# =========================
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not os.path.exists(file_path):
    print("Dataset not found!")
    exit()

df = pd.read_csv(file_path)

# =========================
# CLEANING
# =========================
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.fillna(df.mean(numeric_only=True), inplace=True)

# =========================
# ENCODING
# =========================
encoders = {}

for col in df.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# =========================
# SPLIT
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# SAVE COLUMN ORDER (VERY IMPORTANT)
columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# PREDICTION (THRESHOLD)
# =========================
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_prob > threshold).astype(int)

# =========================
# EVALUATION
# =========================
print("\nMODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# SAVE OUTPUTS
# =========================
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.clf()

# Feature Importance
importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feat_df.plot(kind="bar", x="Feature", y="Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.clf()

# =========================
# SAVE ARTIFACTS
# =========================
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(columns, "models/columns.pkl")

print("\nALL FILES SAVED SUCCESSFULLY")