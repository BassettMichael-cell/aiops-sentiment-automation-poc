import os
import joblib
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Create output folder if it doesn't exist
os.makedirs("models", exist_ok=True)

print("🚀 Starting automated ML training pipeline...")

SAMPLE_SIZE = 1000
SEED = 42

# Load dataset - use the canonical HF dataset id
print("Loading IMDb dataset from Hugging Face (imdb)...")
try:
    ds = load_dataset("imdb")  # no trust_remote_code
    # Shuffle BEFORE sampling so we don't accidentally pull one class
    ds_train = ds["train"].shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    print(f"Successfully loaded {len(ds_train)} examples")
    df = pd.DataFrame(ds_train)

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using fallback dummy data for testing (small sample)")
    data = {
        "text": ["This movie is great!", "This movie is terrible."] * 500,
        "label": [1, 0] * 500
    }
    df = pd.DataFrame(data)

# Quick text preprocessing
df["text"] = df["text"].astype(str).str.lower().str.strip()

X = df["text"]
y = df["label"].astype(int)  # 0 = negative, 1 = positive

# Safety check: ensure we have at least two classes
classes = np.unique(y)
print(f"Label distribution check: {pd.Series(y).value_counts().to_dict()}")
if len(classes) < 2:
    raise ValueError(
        f"Training data contains only one class: {classes}. "
        "Increase sample size or ensure shuffling/stratification."
    )

# Train/test split (stratify to guarantee both classes in train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Vectorize text
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train simple logistic regression model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")

print("\nClassification Report:")
report_text = classification_report(
    y_test, y_pred, target_names=["Negative", "Positive"]
)
print(report_text)

# Save the vectorizer + model together
joblib.dump((vectorizer, model), "models/sentiment_model.joblib")
print("💾 Model and vectorizer saved to models/sentiment_model.joblib")

# Generate a simple markdown report
report = f"""# Training Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Dataset**: IMDb Sentiment ({SAMPLE_SIZE} samples from `imdb`, shuffled)  
**Train/Test split**: 80/20 (stratified)  
**Accuracy**: {acc:.4f}

### Classification Report 
Model saved as `models/sentiment_model.joblib`
"""

with open("models/training_report.md", "w") as f:
    f.write(report)

print("Report saved")
print("Done!")
