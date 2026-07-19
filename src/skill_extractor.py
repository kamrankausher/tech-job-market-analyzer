"""
Skill extraction module.
Extracts skills from job description text using regex matching against the skill taxonomy.
Computes skill frequency rankings overall and by role/experience level.
"""
import pandas as pd
import numpy as np
import re
from typing import List, Set, Dict, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from src.skill_taxonomy import SKILL_PATTERNS, SKILL_TAXONOMY, get_skill_taxonomy


def extract_skills_from_text(text: str) -> Set[str]:
    """
    Extract skills mentioned in a single text string.
    
    Args:
        text: Job description or resume text.
    
    Returns:
        Set of matched skill names.
    """
    if not isinstance(text, str) or not text.strip():
        return set()
    
    matched_skills = set()
    for skill_name, pattern in SKILL_PATTERNS:
        if pattern.search(text):
            matched_skills.add(skill_name)
    
    # Deduplicate overlapping skills:
    # If "Amazon Web Services" matched, also count as "AWS"
    if "Amazon Web Services" in matched_skills:
        matched_skills.add("AWS")
    if "Google Cloud" in matched_skills:
        matched_skills.add("GCP")
    if "Natural Language Processing" in matched_skills:
        matched_skills.add("NLP")
    if "Large Language Models" in matched_skills:
        matched_skills.add("LLM")
    # If "PySpark" matched, also count "Apache Spark"
    if "PySpark" in matched_skills:
        matched_skills.add("Apache Spark")
    # If "React Native" matched, also count "React"
    if "React Native" in matched_skills:
        matched_skills.add("React")
    
    return matched_skills


def extract_skills_batch(df: pd.DataFrame, text_col: str = "description") -> pd.DataFrame:
    """
    Extract skills from all rows in a DataFrame.
    
    Args:
        df: DataFrame with a text column.
        text_col: Name of the column containing text to analyze.
    
    Returns:
        DataFrame with additional 'extracted_skills' column (list of skill names)
        and 'skill_count' column (number of skills found).
    """
    result = df.copy()
    result['extracted_skills'] = result[text_col].apply(
        lambda x: sorted(extract_skills_from_text(x)) if isinstance(x, str) else []
    )
    result['skill_count'] = result['extracted_skills'].apply(len)
    return result


def compute_skill_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall skill frequency rankings from extracted skills.
    
    Args:
        df: DataFrame with 'extracted_skills' column (list of skills per row).
    
    Returns:
        DataFrame with columns: skill, category, count, percentage
        sorted by count descending.
    """
    total_postings = len(df)
    taxonomy = get_skill_taxonomy()
    
    # Count each skill
    skill_counts = {}
    for skills_list in df['extracted_skills']:
        for skill in skills_list:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    # Build result DataFrame
    rows = []
    for skill, count in skill_counts.items():
        rows.append({
            'skill': skill,
            'category': taxonomy.get(skill, 'Other'),
            'count': count,
            'percentage': round(count / total_postings * 100, 2),
        })
    
    result = pd.DataFrame(rows)
    result = result.sort_values('count', ascending=False).reset_index(drop=True)
    result['rank'] = range(1, len(result) + 1)
    return result


def compute_skill_frequencies_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute skill frequencies grouped by experience level.
    
    Args:
        df: DataFrame with 'extracted_skills' and 'formatted_experience_level' columns.
    
    Returns:
        DataFrame with columns: skill, experience_level, count, percentage, category
    """
    taxonomy = get_skill_taxonomy()
    rows = []
    
    for level, group in df.groupby('formatted_experience_level'):
        total = len(group)
        skill_counts = {}
        for skills_list in group['extracted_skills']:
            for skill in skills_list:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        for skill, count in skill_counts.items():
            rows.append({
                'skill': skill,
                'experience_level': level,
                'count': count,
                'percentage': round(count / total * 100, 2),
                'category': taxonomy.get(skill, 'Other'),
            })
    
    return pd.DataFrame(rows).sort_values(['experience_level', 'count'], ascending=[True, False])


def compute_skill_frequencies_by_role(df: pd.DataFrame, top_n_roles: int = 15) -> pd.DataFrame:
    """
    Compute skill frequencies grouped by job title (top N most common titles).
    
    Args:
        df: DataFrame with 'extracted_skills' and 'title' columns.
        top_n_roles: Number of top roles to include.
    
    Returns:
        DataFrame with columns: skill, role, count, percentage, category
    """
    taxonomy = get_skill_taxonomy()
    
    # Normalize titles for grouping
    title_counts = df['title'].value_counts()
    top_titles = title_counts.head(top_n_roles).index.tolist()
    
    rows = []
    for title in top_titles:
        group = df[df['title'] == title]
        total = len(group)
        skill_counts = {}
        for skills_list in group['extracted_skills']:
            for skill in skills_list:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        for skill, count in skill_counts.items():
            rows.append({
                'skill': skill,
                'role': title,
                'count': count,
                'percentage': round(count / total * 100, 2),
                'category': taxonomy.get(skill, 'Other'),
            })
    
    return pd.DataFrame(rows).sort_values(['role', 'count'], ascending=[True, False])


def fit_tfidf(
    texts: pd.Series,
    max_features: int = 5000,
    save_path: Optional[Path] = None
) -> tuple:
    """
    Fit a TF-IDF vectorizer on job description texts.
    
    Args:
        texts: Series of text strings.
        max_features: Maximum number of TF-IDF features.
        save_path: If provided, save the fitted vectorizer to this path.
    
    Returns:
        Tuple of (tfidf_matrix, vectorizer)
    """
    # Clean texts
    clean_texts = texts.fillna("").astype(str)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
    )
    
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, save_path)
        print(f"TF-IDF vectorizer saved to {save_path}")
    
    return tfidf_matrix, vectorizer


def build_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build binary skill-presence features from extracted_skills column.
    
    Args:
        df: DataFrame with 'extracted_skills' column.
    
    Returns:
        DataFrame with binary columns for each skill found in the dataset.
    """
    # Get all unique skills that appear in the data
    all_skills = set()
    for skills_list in df['extracted_skills']:
        all_skills.update(skills_list)
    
    all_skills = sorted(all_skills)
    
    # Build binary matrix
    skill_data = {}
    for skill in all_skills:
        skill_data[f"skill_{skill}"] = df['extracted_skills'].apply(
            lambda x: 1 if skill in x else 0
        )
    
    return pd.DataFrame(skill_data, index=df.index)


def compare_skills_to_market(
    user_skills: Set[str],
    market_demand: pd.DataFrame,
    top_n: int = 15
) -> Dict:
    """
    Compare a user's skills against market demand rankings.
    
    Args:
        user_skills: Set of skill names the user has.
        market_demand: DataFrame from compute_skill_frequencies() with 'skill', 'rank', 'percentage'.
        top_n: Number of top missing skills to return.
    
    Returns:
        Dict with 'matched_skills', 'missing_skills', 'match_score', etc.
    """
    # Normalize user skills for matching
    user_skills_normalized = set()
    for skill in user_skills:
        # Try to find case-insensitive match in taxonomy
        for tax_skill in SKILL_TAXONOMY:
            if skill.lower() == tax_skill.lower():
                user_skills_normalized.add(tax_skill)
                break
    
    market_skills = set(market_demand['skill'].tolist())
    
    # Matched skills (user has + market wants)
    matched = user_skills_normalized & market_skills
    matched_details = market_demand[market_demand['skill'].isin(matched)].sort_values('rank')
    
    # Missing high-demand skills
    missing = market_skills - user_skills_normalized
    missing_details = market_demand[market_demand['skill'].isin(missing)].head(top_n)
    
    # Match score: percentage of top-50 market skills the user has
    top_50_skills = set(market_demand.head(50)['skill'].tolist())
    top_50_matched = user_skills_normalized & top_50_skills
    match_score = len(top_50_matched) / min(len(top_50_skills), 50) * 100 if top_50_skills else 0
    
    return {
        'matched_skills': matched_details.to_dict('records'),
        'missing_skills': missing_details.to_dict('records'),
        'match_score': round(match_score, 1),
        'total_user_skills': len(user_skills_normalized),
        'total_matched': len(matched),
        'total_market_skills': len(market_skills),
    }
