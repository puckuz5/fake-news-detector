import pandas as pd
import numpy as np
import joblib
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report)

# ──────────────────────────────────────────────
# STEP 1: Load dataset
# Change this path to match your CSV filename
# ──────────────────────────────────────────────
DATA_PATH = "data/news.csv"   # ← change if your file has different name

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, encoding='latin-1')

# Print first few rows so you can see what columns exist
print("\nFirst 3 rows of your data:")
print(df.head(3))
print("\nColumns:", df.columns.tolist())
print("Shape:", df.shape)

# ──────────────────────────────────────────────
# STEP 2: Auto-detect columns
# Different datasets use different column names
# This handles the most common ones automatically
# ──────────────────────────────────────────────

text_col     = 'Statement'
headline_col = None
label_col    = 'Label'

print(f"\nDetected columns:")
print(f"  Text column    : {text_col}")
print(f"  Headline column: {headline_col}")
print(f"  Label column   : {label_col}")

# If detection failed, manually set them here
if not label_col:
    print("\nERROR: Could not detect label column.")
    print("Please set label_col manually in the code.")
    print("Available columns:", df.columns.tolist())
    exit()

# ──────────────────────────────────────────────
# STEP 3: Clean and combine text
# We combine headline + text for better accuracy
# ──────────────────────────────────────────────
print("\nCleaning data...")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()                    # lowercase
    text = re.sub(r'http\S+', '', text)         # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)        # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text

# Combine headline and text if both exist
if headline_col and text_col:
    df['combined'] = df[headline_col].fillna('') + ' ' + df[text_col].fillna('')
elif text_col:
    df['combined'] = df[text_col].fillna('')
elif headline_col:
    df['combined'] = df[headline_col].fillna('')
else:
    print("ERROR: No text column found.")
    exit()

df['combined'] = df['combined'].apply(clean_text)

# ──────────────────────────────────────────────
# STEP 4: Prepare labels
# Convert FAKE/REAL to 0/1
# 1 = FAKE, 0 = REAL
# ──────────────────────────────────────────────
print("\nPreparing labels...")
print("Unique label values:", df[label_col].unique())

# Handle different label formats
label_map = {}
for val in df[label_col].unique():
    v = str(val).upper().strip()
    if v in ['FAKE', 'FALSE', '1', 'MISINFORMATION', 'UNRELIABLE', 'FAKE']:
     label_map[val] = 1
    elif v in ['REAL', 'TRUE', '0', 'RELIABLE', 'LEGITIMATE']:
     label_map[val] = 0

if not label_map:
    # If unknown labels, assume binary 0/1 already
    print("Using labels as-is (assuming 0=real, 1=fake)")
    df['label_encoded'] = df[label_col].astype(int)
else:
    df['label_encoded'] = df[label_col].map(label_map)

# Drop rows where label couldn't be mapped
df = df.dropna(subset=['label_encoded', 'combined'])
df = df[df['combined'].str.len() > 10]  # remove very short texts

print(f"Label distribution:")
print(f"  REAL (0): {(df['label_encoded']==0).sum()}")
print(f"  FAKE (1): {(df['label_encoded']==1).sum()}")
print(f"  Total   : {len(df)}")

# ──────────────────────────────────────────────
# STEP 5: Split data
# 80% training, 20% testing
# random_state=42 means results are reproducible
# ──────────────────────────────────────────────
print("\nSplitting data (80% train, 20% test)...")
X = df['combined']
y = df['label_encoded'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # keeps FAKE/REAL ratio same in train and test
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ──────────────────────────────────────────────
# STEP 6: TF-IDF Vectorization
# Converts text into numbers the model can read
# max_features=50000 = use top 50,000 words
# ngram_range=(1,2) = use single words AND pairs
#   e.g. "fake news" as one feature, not just
#   "fake" and "news" separately
# ──────────────────────────────────────────────
print("\nBuilding TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),       # unigrams + bigrams
    stop_words='english',     # ignore common words like "the", "is"
    sublinear_tf=True         # use log(tf) to reduce impact of very common words
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# ──────────────────────────────────────────────
# STEP 7: Train Model 1 — Logistic Regression
# Fast, accurate, and works great with TF-IDF
# C=1.0 is the regularization strength
# ──────────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    n_jobs=-1          # use all CPU cores
)
lr_model.fit(X_train_vec, y_train)

lr_preds = lr_model.predict(X_test_vec)
lr_acc   = accuracy_score(y_test, lr_preds)
lr_f1    = f1_score(y_test, lr_preds, average='weighted')

print(f"\nLogistic Regression Results:")
print(f"  Accuracy : {lr_acc:.4f} ({lr_acc*100:.1f}%)")
print(f"  F1 Score : {lr_f1:.4f}")
print(classification_report(y_test, lr_preds,
      target_names=['REAL', 'FAKE']))

# ──────────────────────────────────────────────
# STEP 8: Train Model 2 — Random Forest
# More complex, sometimes more accurate
# n_estimators=200 = 200 decision trees
# ──────────────────────────────────────────────
print("\nTraining Random Forest (this takes 2-3 minutes)...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_vec, y_train)

rf_preds = rf_model.predict(X_test_vec)
rf_acc   = accuracy_score(y_test, rf_preds)
rf_f1    = f1_score(y_test, rf_preds, average='weighted')

print(f"\nRandom Forest Results:")
print(f"  Accuracy : {rf_acc:.4f} ({rf_acc*100:.1f}%)")
print(f"  F1 Score : {rf_f1:.4f}")
print(classification_report(y_test, rf_preds,
      target_names=['REAL', 'FAKE']))

# ──────────────────────────────────────────────
# STEP 9: Pick the best model and save it
# We save both the model AND the vectorizer
# The app needs both to make predictions
# ──────────────────────────────────────────────
print("\nComparing models...")
if lr_f1 >= rf_f1:
    best_model = lr_model
    best_name  = "Logistic Regression"
    best_f1    = lr_f1
    best_acc   = lr_acc
else:
    best_model = rf_model
    best_name  = "Random Forest"
    best_f1    = rf_f1
    best_acc   = rf_acc

print(f"Best model: {best_name}")
print(f"  Accuracy: {best_acc*100:.1f}%")
print(f"  F1 Score: {best_f1:.4f}")

os.makedirs("model", exist_ok=True)
joblib.dump(best_model,  "model/model.pkl")
joblib.dump(vectorizer,  "model/vectorizer.pkl")

# Save model info for the app to display
model_info = {
    "name":     best_name,
    "accuracy": round(best_acc * 100, 2),
    "f1":       round(best_f1, 4),
    "lr_acc":   round(lr_acc * 100, 2),
    "rf_acc":   round(rf_acc * 100, 2),
}
joblib.dump(model_info, "model/model_info.pkl")

print("\nSaved files:")
print("  model/model.pkl       ← trained model")
print("  model/vectorizer.pkl  ← TF-IDF vectorizer")
print("  model/model_info.pkl  ← accuracy stats")
print("\nTraining complete! Now run: streamlit run app.py")
