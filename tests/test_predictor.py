"""
Tests for predictor module.
Tests model loading and prediction pipeline.
"""
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import ExperienceLevelPredictor

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# Skip if model artifacts don't exist
pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model.joblib").exists(),
    reason="Model artifacts not present (run train_pipeline.py first)"
)


class TestExperienceLevelPredictor:
    """Tests for the predictor class."""
    
    @pytest.fixture
    def predictor(self):
        p = ExperienceLevelPredictor(ARTIFACTS_DIR)
        p.load()
        return p
    
    def test_load_success(self, predictor):
        assert predictor.is_loaded
    
    def test_predict_returns_dict(self, predictor):
        result = predictor.predict("Senior Python developer with 10 years experience in machine learning and AWS")
        assert isinstance(result, dict)
    
    def test_predict_has_required_keys(self, predictor):
        result = predictor.predict("Data analyst skilled in SQL and Tableau")
        assert 'predicted_level' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'extracted_skills' in result
        assert 'skill_count' in result
    
    def test_predict_level_is_valid(self, predictor):
        result = predictor.predict("Junior developer learning Python and JavaScript")
        valid_levels = {'Junior', 'Mid-level', 'Senior', 'Lead', 'Management'}
        assert result['predicted_level'] in valid_levels
    
    def test_confidence_in_range(self, predictor):
        result = predictor.predict("Machine learning engineer with PyTorch and Kubernetes")
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict("Data scientist with deep learning experience")
        total_prob = sum(result['probabilities'].values())
        assert abs(total_prob - 1.0) < 0.01, f"Probabilities should sum to ~1.0, got {total_prob}"
    
    def test_extracts_skills(self, predictor):
        result = predictor.predict("Expert in Python, TensorFlow, and Docker")
        assert "Python" in result['extracted_skills']
        assert "TensorFlow" in result['extracted_skills']
        assert "Docker" in result['extracted_skills']
    
    def test_get_class_names(self, predictor):
        classes = predictor.get_class_names()
        assert len(classes) > 0
        assert isinstance(classes, list)
    
    def test_empty_text(self, predictor):
        result = predictor.predict("")
        assert 'predicted_level' in result  # Should still return a prediction
