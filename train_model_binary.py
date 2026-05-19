import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

FEATURE_FILE = "features_binary.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(FEATURE_FILE)

feature_columns = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "colorfulness",
    "rule_of_thirds_score",
    "symmetry_score",
    "aspect_ratio"
]

X = df[feature_columns]
y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, os.path.join(MODEL_DIR, "composition_model_binary.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_binary.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder_binary.pkl"))

print("\nModel binary berhasil disimpan di folder model/")