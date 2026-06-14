"""
Prediction Module
Loads trained model and makes predictions on resume text
"""

import joblib
import os
import numpy as np
from src.preprocess import clean_text


# Global variables for model and vectorizer
_model = None
_vectorizer = None


def load_model_and_vectorizer():
    """
    Load the trained model and vectorizer from disk
    
    Returns:
        tuple: (model, vectorizer)
    
    Raises:
        FileNotFoundError: If model files are not found
    """
    global _model, _vectorizer
    
    # Return cached model if already loaded
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer
    
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train the model first by running: python src/train.py"
        )
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Vectorizer file not found at {vectorizer_path}. "
            "Please train the model first by running: python src/train.py"
        )
    
    # Load model and vectorizer
    _model = joblib.load(model_path)
    _vectorizer = joblib.load(vectorizer_path)
    
    return _model, _vectorizer


def predict_job(resume_text):
    """
    Predict the most likely job role for a resume
    
    Args:
        resume_text (str): Raw resume text
    
    Returns:
        tuple: (predicted_category, confidence_score)
    """
    model, vectorizer = load_model_and_vectorizer()
    
    # Clean and vectorize text
    cleaned = clean_text(resume_text)
    vector = vectorizer.transform([cleaned])
    
    # Make prediction
    prediction = model.predict(vector)
    probabilities = model.predict_proba(vector)
    
    return prediction[0], probabilities.max()


def predict_top3(resume_text):
    """
    Predict top 3 most likely job roles with confidence scores
    
    Args:
        resume_text (str): Raw resume text
    
    Returns:
        list: List of tuples [(category, probability), ...] sorted by probability descending
              Returns top 3 predictions
    
    Example:
        >>> predict_top3("Experienced Python developer...")
        [('Software Developer', 0.85), ('Data Scientist', 0.10), ('Web Developer', 0.03)]
    """
    if not resume_text or not resume_text.strip():
        return []
    
    try:
        model, vectorizer = load_model_and_vectorizer()
        
        # Clean and vectorize text
        cleaned = clean_text(resume_text)
        
        if not cleaned:
            return []
        
        vector = vectorizer.transform([cleaned])
        
        # Get prediction probabilities for all classes
        probabilities = model.predict_proba(vector)[0]
        
        # Get class labels
        classes = model.classes_
        
        # Create list of (category, probability) tuples
        predictions = list(zip(classes, probabilities))
        
        # Sort by probability in descending order
        predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        # Return top 3
        return predictions_sorted[:3]
    
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")
