"""
Skill Gap Tool Page
User pastes resume text or comma-separated skills.
Compares against real market demand and predicts experience level.
All logic powered by real trained model and extracted data.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.skill_extractor import extract_skills_from_text, compare_skills_to_market
from src.skill_taxonomy import get_skill_taxonomy, get_skill_categories
from src.predictor import ExperienceLevelPredictor
from src.ui_utils import apply_custom_css

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Skill Gap Tool", page_icon="", layout="wide")


@st.cache_data
def load_skill_demand():
    """Load market skill demand data."""
    return pd.read_csv(ARTIFACTS_DIR / "skill_demand.csv")


@st.cache_resource
def load_predictor():
    """Load the trained ML model."""
    predictor = ExperienceLevelPredictor(ARTIFACTS_DIR)
    success = predictor.load()
    return predictor if success else None


def main():
    apply_custom_css()
    st.markdown('<div class="title-glow"> Skill Gap Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 1.1rem; color: #cbd5e1;'>Paste your resume or list your skills to see how you compare "
        "against <b>real market demand</b> from 5,700+ Tech job postings.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Load data
    try:
        skill_demand = load_skill_demand()
    except FileNotFoundError:
        st.error("Missing skill_demand.csv. Run `python scripts/train_pipeline.py` first.")
        return
    
    predictor = load_predictor()
    
    # =========================================================================
    # INPUT
    # =========================================================================
    input_method = st.radio(
        "How would you like to input your skills?",
        ["📝 Paste Resume Text", "📋 Comma-Separated Skills"],
        horizontal=True,
    )
    
    user_text = ""
    user_skills = set()
    
    if input_method == "📝 Paste Resume Text":
        user_text = st.text_area(
            "Paste your resume or profile text below:",
            height=200,
            placeholder="e.g., Experienced data scientist with 3 years of experience in Python, "
                       "machine learning, and SQL. Built recommendation systems using scikit-learn "
                       "and deployed models on AWS using Docker...",
        )
    else:
        skills_input = st.text_input(
            "Enter your skills (comma-separated):",
            placeholder="e.g., Python, SQL, Machine Learning, TensorFlow, AWS, Docker",
        )
        if skills_input:
            user_text = skills_input
            # Also parse as direct skill names
            user_skills = set(s.strip() for s in skills_input.split(",") if s.strip())
    
    analyze_btn = st.button(" Analyze My Skills", type="primary", use_container_width=True)
    
    if analyze_btn and user_text.strip():
        st.markdown("---")
        
        # Extract skills from text
        extracted = extract_skills_from_text(user_text)
        # Combine with directly specified skills
        all_user_skills = extracted | user_skills
        
        if not all_user_skills:
            st.warning("No recognized tech skills found in your input. Try adding more detail or specific skill names.")
            return
        
        # Compare against market
        comparison = compare_skills_to_market(all_user_skills, skill_demand, top_n=15)
        
        # =====================================================================
        # RESULTS
        # =====================================================================
        st.subheader(" Your Skill Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Skills Detected", comparison['total_user_skills'])
        col2.metric("Market Matches", comparison['total_matched'])
        col3.metric("Market Coverage", f"{comparison['match_score']}%")
        col4.metric("Skills Tracked", comparison['total_market_skills'])
        
        st.markdown("---")
        
        # Two-column layout
        left_col, right_col = st.columns(2)
        
        # Matched Skills
        with left_col:
            st.subheader(" Your Matched Skills")
            matched = comparison['matched_skills']
            if matched:
                matched_df = pd.DataFrame(matched)
                matched_df = matched_df[['skill', 'rank', 'percentage', 'category']].rename(columns={
                    'skill': 'Skill',
                    'rank': 'Market Rank',
                    'percentage': 'Demand %',
                    'category': 'Category',
                })
                st.dataframe(matched_df, use_container_width=True, hide_index=True)
            else:
                st.info("No matched skills found.")
        
        # Missing Skills
        with right_col:
            st.subheader("Top Missing High-Demand Skills")
            missing = comparison['missing_skills']
            if missing:
                missing_df = pd.DataFrame(missing)
                missing_df = missing_df[['skill', 'rank', 'percentage', 'category']].rename(columns={
                    'skill': 'Skill',
                    'rank': 'Market Rank',
                    'percentage': 'Demand %',
                    'category': 'Category',
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("You have all the top skills! 🎉")
        
        # Visual Gap Chart
        st.subheader("Skill Gap Visualization")
        
        # Create a combined chart showing what user has vs market top 20
        top_20 = skill_demand.head(20).copy()
        top_20['user_has'] = top_20['skill'].apply(
            lambda s: 'You Have' if s in all_user_skills else 'You\'re Missing'
        )
        
        fig_gap = px.bar(
            top_20,
            x='percentage', y='skill',
            orientation='h',
            color='user_has',
            title='Your Skills vs. Top 20 Market Demands',
            labels={'percentage': '% of Job Postings', 'skill': 'Skill'},
            color_discrete_map={
                'You Have': '#2ecc71',
                "You're Missing": '#e74c3c',
            },
        )
        fig_gap.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            legend_title_text='Status',
        )
        st.plotly_chart(fig_gap, use_container_width=True)
        
        # Category coverage radar chart
        st.subheader("Category Coverage")
        categories = get_skill_categories()
        cat_scores = []
        for cat, skills in categories.items():
            cat_total = len(skills)
            cat_match = len(all_user_skills & set(skills))
            cat_scores.append({
                'Category': cat,
                'Coverage %': round(cat_match / cat_total * 100, 1) if cat_total > 0 else 0,
                'Matched': cat_match,
                'Total': cat_total,
            })
        
        cat_df = pd.DataFrame(cat_scores).sort_values('Coverage %', ascending=False)
        cat_df = cat_df[cat_df['Matched'] > 0]  # Only show categories with at least 1 match
        
        if not cat_df.empty:
            fig_cat = px.bar(
                cat_df,
                x='Coverage %', y='Category',
                orientation='h',
                title='Your Skill Coverage by Category',
                color='Coverage %',
                color_continuous_scale='RdYlGn',
                text='Matched',
            )
            fig_cat.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400,
            )
            fig_cat.update_traces(texttemplate='%{text} matched', textposition='inside')
            st.plotly_chart(fig_cat, use_container_width=True)
        
        # =====================================================================
        # ML PREDICTION
        # =====================================================================
        if predictor and predictor.is_loaded:
            st.markdown("---")
            st.subheader("Experience Level Prediction")
            st.markdown(
                "Based on your skills and text, our model predicts what experience level "
                "your profile best matches:"
            )
            
            prediction = predictor.predict(user_text)
            
            # Display prediction
            pred_col1, pred_col2 = st.columns([1, 2])
            
            with pred_col1:
                st.metric(
                    "Predicted Level",
                    prediction['predicted_level'],
                    help="Based on Logistic Regression trained on 94K+ job postings"
                )
                st.metric(
                    "Confidence",
                    f"{prediction['confidence']*100:.1f}%"
                )
            
            with pred_col2:
                # Probability distribution
                prob_df = pd.DataFrame([
                    {'Level': k, 'Probability': v}
                    for k, v in prediction['probabilities'].items()
                ]).sort_values('Probability', ascending=True)
                
                fig_prob = px.bar(
                    prob_df,
                    x='Probability', y='Level',
                    orientation='h',
                    title='Prediction Probability Distribution',
                    color='Probability',
                    color_continuous_scale='Viridis',
                )
                fig_prob.update_layout(height=300)
                st.plotly_chart(fig_prob, use_container_width=True)
            
            st.caption(
                " This prediction is based on text similarity to job postings at different "
                "experience levels. It's an indicator, not a definitive assessment."
            )
        else:
            st.info(
                "ML prediction unavailable — model artifacts not found. "
                "Run `python scripts/train_pipeline.py` to train the model."
            )
    
    elif analyze_btn:
        st.warning("Please enter some text or skills to analyze.")
    
    # =========================================================================
    # SKILL TAXONOMY REFERENCE
    # =========================================================================
    with st.expander("📚 Full Skill Taxonomy Reference", expanded=False):
        taxonomy = get_skill_taxonomy()
        tax_df = pd.DataFrame([
            {'Skill': skill, 'Category': cat}
            for skill, cat in taxonomy.items()
        ]).sort_values(['Category', 'Skill'])
        
        st.markdown(f"**{len(taxonomy)} skills** tracked across **{tax_df['Category'].nunique()} categories**")
        st.dataframe(tax_df, use_container_width=True, hide_index=True, height=400)


if __name__ == "__main__":
    main()
