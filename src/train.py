"""
Model Training Module
Trains a multi-class classification model for resume job prediction
"""

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_text


def train_model():
    """
    Train the resume classification model
    
    Steps:
    1. Load dataset
    2. Clean text data
    3. Apply TF-IDF vectorization
    4. Train Logistic Regression model
    5. Evaluate performance
    6. Save model and vectorizer
    """
    
    print("=" * 60)
    print("Starting Model Training...")
    print("=" * 60)
    
    # Load dataset
    data_path = os.path.join("data", "final_merged_dataset2.csv")
    print(f"\n1. Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Number of categories: {df['Category'].nunique()}")
    
    # Clean text data
    print("\n2. Cleaning resume text...")
    df["Cleaned"] = df["Resume"].apply(clean_text)
    
    # Prepare features and labels
    X = df["Cleaned"]
    y = df["Category"]
    
    # Apply TF-IDF vectorization
    print("\n3. Applying TF-IDF vectorization (max_features=10000)...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_vec = vectorizer.fit_transform(X)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Split data
    print("\n4. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    # Train model
    print("\n5. Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   Model training completed!")
    
    # Evaluate model
    print("\n6. Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n   Classification Report:")
    print("   " + "-" * 56)
    report = classification_report(y_test, y_pred)
    for line in report.split('\n'):
        print("   " + line)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model and vectorizer
    print("\n7. Saving model and vectorizer...")
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   ✓ Vectorizer saved to: {vectorizer_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
