import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\Ikanabasi Akpan\Downloads\phishing_lightweight.csv")

# IMPORTANT: Use ONLY URL-based features
feature_columns = [
    "url_length",
    "num_dots",
    "has_https",
    "has_at",
    "has_hyphen",
    "has_digits"
]

X = df[feature_columns]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully")
