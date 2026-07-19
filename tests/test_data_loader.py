"""
Tests for data_loader module.
Tests that load functions return expected types and shapes.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_postings, get_dataset_info, filter_tech_roles,
    get_postings_with_experience_level, DATA_DIR,
)


# Skip all tests if data files don't exist
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "ai_jobs_global.csv").exists(),
    reason="Data files not present (download from Kaggle first)"
)


class TestLoadPostings:
    """Tests for the load_postings function."""
    
    def test_returns_dataframe(self):
        df = load_postings()
        assert isinstance(df, pd.DataFrame)
    
    def test_has_required_columns(self):
        df = load_postings()
        required = ['title', 'description', 'location', 'formatted_experience_level']
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_non_empty(self):
        df = load_postings()
        assert len(df) > 0, "DataFrame should not be empty"
    
    def test_row_count_matches_known_dataset(self):
        df = load_postings()
        # The AI Jobs 2026 dataset has 5,773 rows
        assert len(df) == 5773, f"Expected 5,773 rows, got {len(df)}"


class TestGetDatasetInfo:
    """Tests for the get_dataset_info function."""
    
    def test_returns_dict(self):
        df = load_postings()
        info = get_dataset_info(df)
        assert isinstance(info, dict)
    
    def test_has_required_keys(self):
        df = load_postings()
        info = get_dataset_info(df)
        assert 'rows' in info
        assert 'columns' in info
        assert 'null_summary' in info
    
    def test_row_count_accurate(self):
        df = load_postings()
        info = get_dataset_info(df)
        assert info['rows'] == len(df)


class TestFilterTechRoles:
    """Tests for the filter_tech_roles function."""
    
    def test_returns_subset(self):
        df = load_postings()
        tech_df = filter_tech_roles(df)
        assert len(tech_df) < len(df), "Tech filter should reduce row count"
        assert len(tech_df) > 0, "Tech filter should find some tech roles"
    
    def test_contains_known_tech_titles(self):
        df = load_postings()
        tech_df = filter_tech_roles(df)
        titles_lower = tech_df['title'].str.lower()
        # At least some software/data roles should be present
        has_software = titles_lower.str.contains('software').any()
        has_data = titles_lower.str.contains('data').any()
        assert has_software or has_data, "Should contain software or data roles"


class TestGetPostingsWithExperienceLevel:
    """Tests for the get_postings_with_experience_level function."""
    
    def test_no_nulls_in_target(self):
        df = load_postings()
        labeled = get_postings_with_experience_level(df)
        assert labeled['formatted_experience_level'].isna().sum() == 0
    
    def test_valid_levels(self):
        df = load_postings()
        labeled = get_postings_with_experience_level(df)
        valid_levels = {'Junior', 'Mid-level', 'Senior', 'Lead', 'Management'}
        actual_levels = set(labeled['formatted_experience_level'].unique())
        assert actual_levels.issubset(valid_levels), f"Unexpected levels: {actual_levels - valid_levels}"
