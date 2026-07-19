"""
Tech Job Market Skill-Gap Analyzer — Streamlit App
Main entry point. Sets up page config, advanced styling, and sidebar navigation.
"""
import streamlit as st
from pathlib import Path
import sys

# Add src to path so we can import ui_utils
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Tech Job Market Skill-Gap Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui_utils import apply_custom_css

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def check_artifacts():
    """Check if required ML and data artifacts exist."""
    required = [
        "skill_demand.csv",
        "eda_stats.json",
        "model_metrics.json",
        "model.joblib",
        "tfidf_vectorizer.joblib",
    ]
    missing = [f for f in required if not (ARTIFACTS_DIR / f).exists()]
    return missing


def main():
    apply_custom_css()
    
    # Check artifacts
    missing = check_artifacts()
    if missing:
        st.sidebar.warning(
            f" Missing artifacts: {', '.join(missing[:2])}... "
            f"Run `python scripts/train_pipeline.py` to generate them."
        )
    
    # Main page content
    st.markdown('<div class="title-glow">Tech Job Market Skill-Gap Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="feature-card">
            <div class="feature-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="url(#green-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-3d float-anim-delayed"><defs><linearGradient id="green-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#10b981" /><stop offset="100%" stop-color="#3b82f6" /></linearGradient></defs><path d="M18 20V10M12 20V4M6 20V14"></path><polyline points="2 20 22 20"></polyline></svg></div>
            <h3 style="margin-top: 0;">Market Dashboard</h3>
            <p style="color: #cbd5e1;">Explore real skill demand rankings, trends by role 
            and experience level, and market insights derived directly from 
            current 2026 global tech job postings.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="feature-card">
            <div class="feature-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="url(#pink-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-3d float-anim"><defs><linearGradient id="pink-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#ec4899" /><stop offset="100%" stop-color="#8b5cf6" /></linearGradient></defs><circle cx="11" cy="11" r="8"></circle><path d="M21 21L16.65 16.65"></path></svg></div>
            <h3 style="margin-top: 0;">Skill Gap Tool</h3>
            <p style="color: #cbd5e1;">Paste your resume or list your skills to get a 
            personalized gap analysis against real market demand. 
            Discover exactly which high-value skills you are missing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            """
            <div class="feature-card">
            <div class="feature-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="url(#orange-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-3d float-anim-delayed"><defs><linearGradient id="orange-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#f59e0b" /><stop offset="100%" stop-color="#ef4444" /></linearGradient></defs><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><path d="M8 16h.01M16 16h.01"></path></svg></div>
            <h3 style="margin-top: 0;">ML Inference</h3>
            <p style="color: #cbd5e1;">Our Logistic Regression classification model predicts the expected 
            experience level of a job description by extracting TF-IDF and skill features.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    st.markdown(
        """
        ### Architecture & Overview
        Welcome to the **Next-Generation Tech Intelligence Platform**. This application is powered by an advanced data pipeline analyzing **5,700+ recent global tech job postings** from 2026 to deliver real-time market insights.
        
        - **NLP Skill Extraction:** Job descriptions are processed using regular expressions against a custom taxonomy of over 200 modern technologies.
        - **Machine Learning:** A Logistic Regression model predicts job seniority (Junior, Mid-level, Senior, Lead, Management) using TF-IDF vectorized text and binary skill features.
        - **Dashboard UI:** Built with Plotly and Streamlit, featuring custom CSS keyframe animations and glassmorphism styling.
        
        👈 **Use the sidebar** to navigate to the Market Dashboard or the Skill Gap Tool.
        """
    )


if __name__ == "__main__":
    main()
