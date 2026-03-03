import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\Ikanabasi Akpan\Downloads\phishing_lightweight.csv")

suspicious_words = ["login", "verify", "secure", "update", "bank", "account"]

def has_suspicious(url):
    return int(any(word in str(url).lower() for word in suspicious_words))

# Create new column
df["has_suspicious_word"] = df["url_name"].apply(has_suspicious)

feature_columns = [
    "url_length",
    "num_dots",
    "has_https",
    "has_at",
    "has_hyphen",
    "has_digits",
    "has_suspicious_word"   
]

X = df[feature_columns]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (balanced to reduce bias)
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")





