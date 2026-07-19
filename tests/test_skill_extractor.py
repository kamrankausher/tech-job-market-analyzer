"""
Tests for skill_extractor module.
Tests skill extraction on known text snippets and frequency computation.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.skill_extractor import (
    extract_skills_from_text,
    extract_skills_batch,
    compute_skill_frequencies,
    compare_skills_to_market,
)
from src.skill_taxonomy import get_skill_taxonomy, get_skill_count


class TestExtractSkillsFromText:
    """Tests for single-text skill extraction."""
    
    def test_extracts_python(self):
        text = "We need a developer with 3 years of Python experience."
        skills = extract_skills_from_text(text)
        assert "Python" in skills
    
    def test_extracts_multiple_skills(self):
        text = "Required: Python, SQL, Machine Learning, TensorFlow, and AWS experience."
        skills = extract_skills_from_text(text)
        assert "Python" in skills
        assert "SQL" in skills
        assert "TensorFlow" in skills
        assert "AWS" in skills
    
    def test_case_insensitive(self):
        text = "Experience with PYTHON, postgresql, and Docker is required."
        skills = extract_skills_from_text(text)
        assert "Python" in skills
        assert "PostgreSQL" in skills
        assert "Docker" in skills
    
    def test_empty_string(self):
        skills = extract_skills_from_text("")
        assert len(skills) == 0
    
    def test_none_input(self):
        skills = extract_skills_from_text(None)
        assert len(skills) == 0
    
    def test_no_skills_in_text(self):
        text = "Looking for a friendly person who likes working with teams."
        skills = extract_skills_from_text(text)
        # May match "Agile" or other soft skills, but should not match
        # programming languages on generic text
        assert "Python" not in skills
        assert "Java" not in skills
    
    def test_multi_word_skills(self):
        text = "Experience in machine learning and deep learning models."
        skills = extract_skills_from_text(text)
        assert "Machine Learning" in skills
        assert "Deep Learning" in skills
    
    def test_sklearn_variations(self):
        text = "Proficient with scikit-learn and sklearn for model building."
        skills = extract_skills_from_text(text)
        assert "scikit-learn" in skills
    
    def test_cpp(self):
        text = "Strong C++ programming skills required."
        skills = extract_skills_from_text(text)
        assert "C++" in skills
    
    def test_cloud_platforms(self):
        text = "Deploy on AWS, Azure, or Google Cloud Platform."
        skills = extract_skills_from_text(text)
        assert "AWS" in skills
        assert "Azure" in skills
        assert "Google Cloud" in skills


class TestExtractSkillsBatch:
    """Tests for batch skill extraction on DataFrames."""
    
    def test_adds_columns(self):
        df = pd.DataFrame({
            'description': [
                "Python and SQL developer",
                "Java Spring Boot engineer",
                "Data analyst with Tableau",
            ]
        })
        result = extract_skills_batch(df, text_col='description')
        assert 'extracted_skills' in result.columns
        assert 'skill_count' in result.columns
    
    def test_skill_counts_accurate(self):
        df = pd.DataFrame({
            'description': [
                "Python developer",
                "",
            ]
        })
        result = extract_skills_batch(df, text_col='description')
        assert result.iloc[0]['skill_count'] >= 1
        assert result.iloc[1]['skill_count'] == 0


class TestComputeSkillFrequencies:
    """Tests for skill frequency computation."""
    
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            'extracted_skills': [
                ['Python', 'SQL'],
                ['Python', 'AWS'],
                ['SQL', 'Docker'],
            ]
        })
        freq = compute_skill_frequencies(df)
        assert isinstance(freq, pd.DataFrame)
    
    def test_sorted_by_count(self):
        df = pd.DataFrame({
            'extracted_skills': [
                ['Python', 'SQL'],
                ['Python', 'AWS'],
                ['SQL', 'Docker'],
            ]
        })
        freq = compute_skill_frequencies(df)
        counts = freq['count'].tolist()
        assert counts == sorted(counts, reverse=True)
    
    def test_has_required_columns(self):
        df = pd.DataFrame({
            'extracted_skills': [['Python', 'SQL']]
        })
        freq = compute_skill_frequencies(df)
        assert 'skill' in freq.columns
        assert 'count' in freq.columns
        assert 'percentage' in freq.columns
        assert 'rank' in freq.columns


class TestCompareSkillsToMarket:
    """Tests for the skill gap comparison function."""
    
    def test_returns_expected_keys(self):
        market = pd.DataFrame({
            'skill': ['Python', 'SQL', 'AWS'],
            'rank': [1, 2, 3],
            'percentage': [50.0, 40.0, 30.0],
            'count': [500, 400, 300],
            'category': ['Programming Languages', 'Programming Languages', 'Cloud Platforms'],
        })
        result = compare_skills_to_market({'Python', 'Docker'}, market)
        assert 'matched_skills' in result
        assert 'missing_skills' in result
        assert 'match_score' in result
    
    def test_identifies_matched_skills(self):
        market = pd.DataFrame({
            'skill': ['Python', 'SQL', 'AWS'],
            'rank': [1, 2, 3],
            'percentage': [50.0, 40.0, 30.0],
            'count': [500, 400, 300],
            'category': ['PL', 'PL', 'Cloud'],
        })
        result = compare_skills_to_market({'Python', 'SQL'}, market)
        matched_names = [m['skill'] for m in result['matched_skills']]
        assert 'Python' in matched_names
        assert 'SQL' in matched_names


class TestSkillTaxonomy:
    """Tests for the skill taxonomy."""
    
    def test_taxonomy_size(self):
        count = get_skill_count()
        assert count >= 150, f"Taxonomy should have >= 150 skills, got {count}"
        assert count <= 300, f"Taxonomy should have <= 300 skills, got {count}"
    
    def test_has_key_skills(self):
        taxonomy = get_skill_taxonomy()
        essential = ['Python', 'SQL', 'AWS', 'Docker', 'TensorFlow', 'Machine Learning']
        for skill in essential:
            assert skill in taxonomy, f"Missing essential skill: {skill}"
