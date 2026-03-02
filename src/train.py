import os
import joblib
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create folders
os.makedirs("models", exist_ok=True)

print("🚀 Starting automated ML training pipeline...")

# Load small subsample for speed (perfect for PoC)
print("Loading IMDb dataset...")
dataset = load_dataset("imdb", split="train[:1000]")
df = pd.DataFrame(dataset)

# Quick preprocess
df["text"] = df["text"].str.lower().str.strip()

X = df["text"]
y = df["label"]  # 0 = negative, 1 = positive

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump((vectorizer, model), "models/sentiment_model.joblib")
print("💾 Model saved to models/sentiment_model.joblib")

# Generate nice report
report = f"""# Training Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}

**Dataset**: stanfordnlp/imdb", split="train[:1000]
**Train/Test split**: 80/20  
**Accuracy**: {acc:.4f}

### Classification Report
{classification_report(y_test, y_pred)}

Model & vectorizer saved as `sentiment_model.joblib`
"""

with open("models/training_report.md", "w") as f:
    f.write(report)

print("📄 Report saved to models/training_report.md")
print("🎉 Pipeline completed successfully!")
