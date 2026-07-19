"""
Data loading and validation module.
Loads the LinkedIn Job Postings dataset from local CSV files.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime


# Default data directory (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"


def load_postings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load the job postings dataset.
    Prioritizes the 2026 'ai_jobs_global.csv' if available, otherwise falls back
    to the 2024 'postings.csv'.
    """
    ai_jobs_path = data_dir / "ai_jobs_global.csv"
    if ai_jobs_path.exists():
        df = pd.read_csv(ai_jobs_path, low_memory=False)
        # Map columns to expected standard schema
        df = df.rename(columns={
            'job_title': 'title',
            'job_description': 'description',
            'experience_level': 'formatted_experience_level'
        })
        # Create a location column from city and country
        if 'city' in df.columns and 'country' in df.columns:
            df['location'] = df['city'] + ", " + df['country']
        elif 'country' in df.columns:
            df['location'] = df['country']
        else:
            df['location'] = "Unknown"
        return df

    # Fallback to older dataset
    filepath = data_dir / "postings.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"No valid dataset found in {data_dir}. "
            f"Please download ai_jobs_global.csv or postings.csv."
        )
    df = pd.read_csv(filepath, low_memory=False)
    # Basic validation
    assert len(df) > 0, "postings.csv is empty"
    assert "description" in df.columns, "Missing 'description' column"
    assert "title" in df.columns, "Missing 'title' column"
    return df


def load_salaries(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load salaries.csv."""
    filepath = data_dir / "salaries.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"salaries.csv not found at {filepath}")
    return pd.read_csv(filepath)


def load_companies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load companies.csv."""
    filepath = data_dir / "companies.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"companies.csv not found at {filepath}")
    return pd.read_csv(filepath, low_memory=False)


def load_job_industries(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load job_industries.csv."""
    filepath = data_dir / "job_industries.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"job_industries.csv not found at {filepath}")
    return pd.read_csv(filepath)


def load_industries(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load industries.csv (lookup table)."""
    filepath = data_dir / "industries.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"industries.csv not found at {filepath}")
    return pd.read_csv(filepath)


def load_job_skills(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load job_skills.csv (LinkedIn's broad skill categories)."""
    filepath = data_dir / "job_skills.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"job_skills.csv not found at {filepath}")
    return pd.read_csv(filepath)


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return a summary dict with row/col counts, null percentages, dtypes.
    """
    null_info = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round(null_count / len(df) * 100, 2)
        null_info[col] = {"null_count": null_count, "null_pct": null_pct}
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_summary": null_info,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }


def filter_tech_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter postings to tech/data/ML/AI relevant job titles.
    Uses keyword matching on the 'title' column.
    
    Returns:
        Filtered DataFrame containing only tech-relevant postings.
    """
    tech_keywords = [
        'software', 'developer', 'engineer', 'data', 'analyst', 'scientist',
        'machine learning', 'ml ', ' ml', 'deep learning', 'ai ', ' ai',
        'artificial intelligence', 'devops', 'cloud', 'backend', 'frontend',
        'full stack', 'fullstack', 'python', 'java', 'javascript',
        'database', 'dba', 'systems', 'network', 'security', 'cyber',
        'it ', ' it ', 'information technology', 'technical', 'product manager',
        'scrum', 'agile', 'qa', 'quality assurance', 'test', 'automation',
        'web', 'mobile', 'ios', 'android', 'react', 'node',
        'business analyst', 'business intelligence', 'bi ', ' bi',
        'etl', 'data warehouse', 'analytics', 'visualization',
        'ux', 'ui ', ' ui', 'design', 'architect',
        'sre', 'reliability', 'infrastructure', 'platform',
        'nlp', 'computer vision', 'robotics',
    ]
    
    title_lower = df['title'].str.lower()
    mask = title_lower.str.contains('|'.join(tech_keywords), na=False, regex=True)
    return df[mask].copy()


def get_postings_with_experience_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to rows that have a non-null formatted_experience_level.
    These are the rows usable for the ML classification task.
    """
    return df[df['formatted_experience_level'].notna()].copy()
