import pandas as pd
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


DATA_PATH = "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)

drop_cols = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

df.drop(columns=drop_cols, inplace=True)

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "roc_auc": round(roc_auc * 100, 2),
    "confusion_matrix": cm.tolist()
}

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/attrition_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")
joblib.dump(feature_importance, "model/feature_importance.pkl")

with open("model/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel Training Completed")
print("Accuracy:", metrics["accuracy"], "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel files saved successfully in model folder")