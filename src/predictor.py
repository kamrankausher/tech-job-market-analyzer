"""
Predictor module.
Loads the trained model and makes predictions at runtime (used by Streamlit app).
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
from scipy.sparse import hstack, csr_matrix

from src.skill_extractor import extract_skills_from_text
from src.skill_taxonomy import SKILL_TAXONOMY


ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


class ExperienceLevelPredictor:
    """
    Loads a trained model and predicts experience level from text input.
    """
    
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self._model = None
        self._vectorizer = None
        self._label_encoder = None
        self._feature_names = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load model artifacts. Returns True if successful."""
        try:
            self._model = joblib.load(self.artifacts_dir / "model.joblib")
            self._vectorizer = joblib.load(self.artifacts_dir / "tfidf_vectorizer.joblib")
            self._label_encoder = joblib.load(self.artifacts_dir / "label_encoder.joblib")
            self._feature_names = joblib.load(self.artifacts_dir / "feature_names.joblib")
            self._loaded = True
            return True
        except FileNotFoundError as e:
            print(f"Model artifacts not found: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def predict(self, text: str) -> Dict:
        """
        Predict experience level from resume/job description text.
        
        Args:
            text: Resume text or skill description.
        
        Returns:
            Dict with predicted_level, confidence, probabilities, and extracted_skills.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Extract skills
        extracted_skills = extract_skills_from_text(text)
        
        # Build TF-IDF features
        tfidf_features = self._vectorizer.transform([text])
        
        # Build skill features (same structure as training)
        skill_feature_names = [n for n in self._feature_names if n.startswith("skill_")]
        skill_vector = []
        for feat_name in skill_feature_names:
            skill_name = feat_name.replace("skill_", "", 1)
            skill_vector.append(1 if skill_name in extracted_skills else 0)
        
        skill_matrix = csr_matrix([skill_vector])
        
        # Combine features
        X = hstack([tfidf_features, skill_matrix])
        
        # Predict
        predicted_idx = self._model.predict(X)[0]
        probabilities = self._model.predict_proba(X)[0]
        
        predicted_level = self._label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx])
        
        # Build probability dict
        prob_dict = {}
        for idx, class_name in enumerate(self._label_encoder.classes_):
            prob_dict[class_name] = round(float(probabilities[idx]), 4)
        
        return {
            'predicted_level': predicted_level,
            'confidence': round(confidence, 4),
            'probabilities': prob_dict,
            'extracted_skills': sorted(extracted_skills),
            'skill_count': len(extracted_skills),
        }
    
    def get_class_names(self) -> List[str]:
        """Return the list of experience level classes."""
        if not self._loaded:
            return []
        return list(self._label_encoder.classes_)
