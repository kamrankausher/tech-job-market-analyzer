"""
EDA (Exploratory Data Analysis) module.
All functions take real DataFrames and return Plotly figures or summary dicts.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
from datetime import datetime


def plot_role_distribution(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of top N most common job titles."""
    title_counts = df['title'].value_counts().head(top_n).reset_index()
    title_counts.columns = ['title', 'count']
    
    fig = px.bar(
        title_counts,
        x='count', y='title',
        orientation='h',
        title=f'Top {top_n} Most Common Job Titles',
        labels={'count': 'Number of Postings', 'title': 'Job Title'},
        color='count',
        color_continuous_scale='Viridis',
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    return fig


def plot_experience_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of experience level distribution."""
    exp_counts = df['formatted_experience_level'].value_counts().reset_index()
    exp_counts.columns = ['level', 'count']
    
    fig = px.pie(
        exp_counts,
        values='count', names='level',
        title='Experience Level Distribution',
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plot_location_distribution(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of top locations."""
    loc_counts = df['location'].value_counts().head(top_n).reset_index()
    loc_counts.columns = ['location', 'count']
    
    fig = px.bar(
        loc_counts,
        x='count', y='location',
        orientation='h',
        title=f'Top {top_n} Job Locations',
        labels={'count': 'Number of Postings', 'location': 'Location'},
        color='count',
        color_continuous_scale='Tealgrn',
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Bar chart of missing value percentages per column."""
    null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    null_pcts = null_pcts[null_pcts > 0]  # Only show columns with nulls
    
    null_df = null_pcts.reset_index()
    null_df.columns = ['column', 'missing_pct']
    
    fig = px.bar(
        null_df,
        x='missing_pct', y='column',
        orientation='h',
        title='Missing Values by Column (%)',
        labels={'missing_pct': 'Missing %', 'column': 'Column'},
        color='missing_pct',
        color_continuous_scale='Reds',
    )
    fig.update_layout(height=600)
    return fig


def plot_work_type_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of work type distribution."""
    wt_counts = df['formatted_work_type'].value_counts().reset_index()
    wt_counts.columns = ['work_type', 'count']
    
    fig = px.bar(
        wt_counts,
        x='work_type', y='count',
        title='Work Type Distribution',
        labels={'count': 'Number of Postings', 'work_type': 'Work Type'},
        color='work_type',
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    return fig


def plot_posting_trends(df: pd.DataFrame) -> go.Figure:
    """Time series of posting volume by month (if date columns exist)."""
    if 'listed_time' not in df.columns:
        return None
    
    # Convert epoch ms to datetime
    df_copy = df.copy()
    df_copy['listed_date'] = pd.to_datetime(df_copy['listed_time'], unit='ms', errors='coerce')
    df_copy['listed_month'] = df_copy['listed_date'].dt.to_period('M').astype(str)
    
    monthly_counts = df_copy.groupby('listed_month').size().reset_index(name='count')
    monthly_counts = monthly_counts.sort_values('listed_month')
    
    fig = px.line(
        monthly_counts,
        x='listed_month', y='count',
        title='Job Postings Over Time (Monthly)',
        labels={'listed_month': 'Month', 'count': 'Number of Postings'},
        markers=True,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_skill_demand(skill_freq_df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart of top N in-demand skills."""
    top_skills = skill_freq_df.head(top_n)
    
    fig = px.bar(
        top_skills,
        x='percentage', y='skill',
        orientation='h',
        title=f'Top {top_n} Most In-Demand Skills',
        labels={'percentage': '% of Job Postings', 'skill': 'Skill'},
        color='category',
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=700,
        legend_title_text='Category',
    )
    return fig


def plot_skill_demand_by_level(skill_by_level_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Grouped bar chart showing top skills per experience level."""
    # Get top N skills per level
    top_per_level = (
        skill_by_level_df
        .groupby('experience_level')
        .apply(lambda x: x.nlargest(top_n, 'count'), include_groups=False)
        .reset_index(drop=True)
    )
    
    fig = px.bar(
        top_per_level,
        x='skill', y='percentage',
        color='experience_level',
        barmode='group',
        title=f'Top {top_n} Skills by Experience Level',
        labels={'percentage': '% of Postings', 'skill': 'Skill'},
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    return fig


def plot_skill_categories(skill_freq_df: pd.DataFrame) -> go.Figure:
    """Treemap of skills grouped by category."""
    category_counts = skill_freq_df.groupby('category')['count'].sum().reset_index()
    category_counts = category_counts.sort_values('count', ascending=False)
    
    fig = px.treemap(
        skill_freq_df,
        path=['category', 'skill'],
        values='count',
        title='Skill Categories Breakdown',
        color='count',
        color_continuous_scale='Viridis',
    )
    fig.update_layout(height=600)
    return fig


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute EDA summary statistics from the postings DataFrame.
    Returns a dict with real computed values.
    """
    stats = {
        "total_postings": int(len(df)),
        "unique_titles": int(df['title'].nunique()),
        "unique_companies": int(df['company_name'].nunique()) if 'company_name' in df.columns else 0,
        "unique_locations": int(df['location'].nunique()),
        "date_range": {},
        "experience_level_distribution": {},
        "work_type_distribution": {},
        "description_stats": {},
        "missing_value_summary": {},
    }
    
    # Date range from listed_time
    if 'listed_time' in df.columns:
        dates = pd.to_datetime(df['listed_time'], unit='ms', errors='coerce')
        stats["date_range"] = {
            "earliest": str(dates.min()),
            "latest": str(dates.max()),
        }
    
    # Experience level
    if 'formatted_experience_level' in df.columns:
        exp_dist = df['formatted_experience_level'].value_counts(dropna=False).to_dict()
        stats["experience_level_distribution"] = {
            str(k) if k is not None else "Missing": int(v) 
            for k, v in exp_dist.items()
        }
    
    # Work type
    if 'formatted_work_type' in df.columns:
        wt_dist = df['formatted_work_type'].value_counts().to_dict()
        stats["work_type_distribution"] = {str(k): int(v) for k, v in wt_dist.items()}
    
    # Description stats
    if 'description' in df.columns:
        desc_lens = df['description'].dropna().str.len()
        stats["description_stats"] = {
            "non_null_count": int(df['description'].notna().sum()),
            "avg_length": round(float(desc_lens.mean()), 0),
            "median_length": round(float(desc_lens.median()), 0),
            "min_length": int(desc_lens.min()),
            "max_length": int(desc_lens.max()),
        }
    
    # Missing values summary
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            stats["missing_value_summary"][col] = {
                "count": null_count,
                "percentage": round(null_count / len(df) * 100, 2),
            }
    
    return stats
