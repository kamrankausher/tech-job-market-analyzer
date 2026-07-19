"""
Market Dashboard Page
Shows real skill demand rankings, EDA charts, and model metrics.
All data loaded from precomputed artifacts — no hardcoded values.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.ui_utils import apply_custom_css

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Market Dashboard", page_icon="", layout="wide")


@st.cache_data
def load_skill_demand():
    """Load overall skill demand rankings."""
    return pd.read_csv(ARTIFACTS_DIR / "skill_demand.csv")


@st.cache_data
def load_tech_skill_demand():
    """Load tech-role specific skill demand rankings."""
    path = ARTIFACTS_DIR / "tech_skill_demand.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_skill_by_level():
    """Load skill demand by experience level."""
    path = ARTIFACTS_DIR / "skill_demand_by_level.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_skill_by_role():
    """Load skill demand by role."""
    path = ARTIFACTS_DIR / "skill_demand_by_role.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_eda_stats():
    """Load EDA summary statistics."""
    with open(ARTIFACTS_DIR / "eda_stats.json", 'r') as f:
        return json.load(f)


@st.cache_data
def load_model_metrics():
    """Load model performance metrics."""
    with open(ARTIFACTS_DIR / "model_metrics.json", 'r') as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    """Load feature importance data."""
    return pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")


def main():
    apply_custom_css()
    st.markdown('<div class="title-glow"> Market Dashboard</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #cbd5e1;'>Real insights derived from <b>5,700+ Global Tech Job Postings</b> (2026)</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load all data
    try:
        skill_demand = load_skill_demand()
        eda_stats = load_eda_stats()
        model_metrics = load_model_metrics()
        feature_importance = load_feature_importance()
    except FileNotFoundError as e:
        st.error(f"Missing artifact files. Run `python scripts/train_pipeline.py` first.\n\nError: {e}")
        return
    
    # =========================================================================
    # KEY METRICS ROW
    # =========================================================================
    st.markdown('<div class="subtitle-glow">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Postings", f"{eda_stats['total_postings']:,}")
    col2.metric("Unique Titles", f"{eda_stats['unique_titles']:,}")
    col3.metric("Unique Locations", f"{eda_stats['unique_locations']:,}")
    col4.metric("Skills Tracked", f"{len(skill_demand):,}")
    
    st.markdown("---")
    
    # =========================================================================
    # TABS
    # =========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Skill Rankings", "📈 By Experience Level", "💼 By Role", "🤖 Model Performance"
    ])
    
    # TAB 1: Overall Skill Rankings
    with tab1:
        st.markdown('<div class="subtitle-glow">Top In-Demand Skills (All Postings)</div>', unsafe_allow_html=True)
        
        col_filter, col_n = st.columns([2, 1])
        with col_filter:
            categories = ["All"] + sorted(skill_demand['category'].unique().tolist())
            selected_cat = st.selectbox("Filter by Category", categories, key="cat_filter")
        with col_n:
            top_n = st.slider("Number of skills", 10, 50, 25, key="top_n_slider")
        
        filtered = skill_demand.copy()
        if selected_cat != "All":
            filtered = filtered[filtered['category'] == selected_cat]
        
        top_skills = filtered.head(top_n)
        
        fig = px.bar(
            top_skills,
            x='percentage', y='skill',
            orientation='h',
            color='category',
            title=f"Top {top_n} Skills — {'All Categories' if selected_cat == 'All' else selected_cat}",
            labels={'percentage': '% of Job Postings', 'skill': 'Skill'},
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, top_n * 25),
            legend_title_text='Category',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tech-specific view
        tech_demand = load_tech_skill_demand()
        if tech_demand is not None:
            st.markdown('<div class="subtitle-glow">Skills Demand — Tech Roles Only</div>', unsafe_allow_html=True)
            tech_top = tech_demand.head(top_n)
            fig_tech = px.bar(
                tech_top,
                x='percentage', y='skill',
                orientation='h',
                color='category',
                title=f"Top {top_n} Skills in Tech Roles",
                labels={'percentage': '% of Tech Postings', 'skill': 'Skill'},
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_tech.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, top_n * 25),
            )
            st.plotly_chart(fig_tech, use_container_width=True)
        
        # Skill category treemap
        st.markdown('<div class="subtitle-glow">Skill Categories Overview</div>', unsafe_allow_html=True)
        cat_summary = skill_demand.groupby('category')['count'].sum().reset_index()
        cat_summary = cat_summary.sort_values('count', ascending=False)
        fig_tree = px.treemap(
            skill_demand.head(60),
            path=['category', 'skill'],
            values='count',
            title='Skill Categories Breakdown (Top 60 Skills)',
            color='count',
            color_continuous_scale='Viridis',
        )
        fig_tree.update_layout(height=500)
        st.plotly_chart(fig_tree, use_container_width=True)
    
    # TAB 2: By Experience Level
    with tab2:
        st.markdown('<div class="subtitle-glow">Skill Demand by Experience Level</div>', unsafe_allow_html=True)
        
        skill_by_level = load_skill_by_level()
        if skill_by_level is not None:
            levels = sorted(skill_by_level['experience_level'].unique().tolist())
            selected_levels = st.multiselect(
                "Select Experience Levels",
                levels,
                default=levels[:3] if len(levels) >= 3 else levels,
                key="level_select"
            )
            
            if selected_levels:
                level_top_n = st.slider("Skills per level", 5, 20, 10, key="level_top_n")
                
                filtered_level = skill_by_level[skill_by_level['experience_level'].isin(selected_levels)]
                
                # Get top N per level
                top_per_level = (
                    filtered_level
                    .groupby('experience_level')
                    .apply(lambda x: x.nlargest(level_top_n, 'count'), include_groups=False)
                    .reset_index(level=0)
                    .reset_index(drop=True)
                )
                
                fig_level = px.bar(
                    top_per_level,
                    x='skill', y='percentage',
                    color='experience_level',
                    barmode='group',
                    title=f'Top {level_top_n} Skills by Experience Level',
                    labels={'percentage': '% of Postings', 'skill': 'Skill'},
                )
                fig_level.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig_level, use_container_width=True)
                
                # Heatmap
                st.markdown('<div class="subtitle-glow">Skills × Experience Level Heatmap</div>', unsafe_allow_html=True)
                # Get top 20 skills overall for the heatmap
                top_20_skills = skill_demand.head(20)['skill'].tolist()
                heatmap_data = filtered_level[filtered_level['skill'].isin(top_20_skills)]
                
                if not heatmap_data.empty:
                    pivot = heatmap_data.pivot_table(
                        index='skill', columns='experience_level',
                        values='percentage', fill_value=0
                    )
                    fig_heat = px.imshow(
                        pivot,
                        labels=dict(x="Experience Level", y="Skill", color="% of Postings"),
                        title="Top 20 Skills × Experience Level",
                        color_continuous_scale="YlOrRd",
                        aspect="auto",
                    )
                    fig_heat.update_layout(height=600)
                    st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Skill-by-level data not available. Run the training pipeline first.")
    
    # TAB 3: By Role
    with tab3:
        st.markdown('<div class="subtitle-glow">Skill Demand by Job Title</div>', unsafe_allow_html=True)
        
        skill_by_role = load_skill_by_role()
        if skill_by_role is not None:
            roles = sorted(skill_by_role['role'].unique().tolist())
            selected_role = st.selectbox("Select a Role", roles, key="role_select")
            
            if selected_role:
                role_data = skill_by_role[skill_by_role['role'] == selected_role]
                role_top = role_data.nlargest(20, 'count')
                
                fig_role = px.bar(
                    role_top,
                    x='percentage', y='skill',
                    orientation='h',
                    color='category',
                    title=f'Top Skills for "{selected_role}"',
                    labels={'percentage': '% of Postings', 'skill': 'Skill'},
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_role.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=500,
                )
                st.plotly_chart(fig_role, use_container_width=True)
        else:
            st.info("Skill-by-role data not available. Run the training pipeline first.")
    
    # TAB 4: Model Performance
    with tab4:
        st.markdown('<div class="subtitle-glow">Experience Level Classifier — Real Metrics</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        **Model:** {model_metrics['model_type']}  
        **Target:** `{model_metrics['target_variable']}`  
        **Classes:** {', '.join(model_metrics['class_names'])}  
        **Training set:** {model_metrics['dataset_size']['train_size']:,} samples  
        **Test set:** {model_metrics['dataset_size']['test_size']:,} samples  
        **Features:** {model_metrics['dataset_size']['n_features']:,} (TF-IDF + skill features)
        """)
        
        # Test metrics
        st.markdown('<div class="subtitle-glow">Test Set Metrics</div>', unsafe_allow_html=True)
        test = model_metrics['test_metrics']
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Accuracy", f"{test['accuracy']:.4f}")
        mcol2.metric("Precision (macro)", f"{test['precision_macro']:.4f}")
        mcol3.metric("Recall (macro)", f"{test['recall_macro']:.4f}")
        mcol4.metric("F1 Score (macro)", f"{test['f1_macro']:.4f}")
        if test.get('roc_auc_macro'):
            mcol5.metric("ROC-AUC (macro)", f"{test['roc_auc_macro']:.4f}")
        
        # Cross-validation
        cv = model_metrics['cross_validation']
        st.markdown('<div class="subtitle-glow">Cross-Validation (5-Fold)</div>', unsafe_allow_html=True)
        st.markdown(f"**Mean Accuracy:** {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}")
        
        cv_df = pd.DataFrame({
            'Fold': [f"Fold {i+1}" for i in range(len(cv['fold_scores']))],
            'Accuracy': cv['fold_scores']
        })
        fig_cv = px.bar(
            cv_df, x='Fold', y='Accuracy',
            title='Cross-Validation Scores',
            color='Accuracy',
            color_continuous_scale='Greens',
        )
        fig_cv.update_layout(height=300)
        st.plotly_chart(fig_cv, use_container_width=True)
        
        # Per-class metrics
        st.markdown('<div class="subtitle-glow">Per-Class Performance</div>', unsafe_allow_html=True)
        per_class = model_metrics['per_class_report']
        pc_df = pd.DataFrame(per_class).T.reset_index()
        pc_df.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        st.dataframe(pc_df, use_container_width=True, hide_index=True)
        
        # Confusion matrix
        st.markdown('<div class="subtitle-glow">Confusion Matrix</div>', unsafe_allow_html=True)
        conf_matrix = model_metrics['confusion_matrix']
        fig_cm = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=model_metrics['class_names'],
            y=model_metrics['class_names'],
            title='Confusion Matrix',
            color_continuous_scale='Blues',
            text_auto=True,
        )
        fig_cm.update_layout(height=500)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature Importance
        st.markdown('<div class="subtitle-glow">Top Feature Importances (Permutation)</div>', unsafe_allow_html=True)
        fig_fi = px.bar(
            feature_importance.head(20),
            x='importance_mean', y='feature',
            orientation='h',
            title='Top 20 Features by Permutation Importance',
            labels={'importance_mean': 'Importance (accuracy drop)', 'feature': 'Feature'},
            error_x='importance_std',
            color='importance_mean',
            color_continuous_scale='Plasma',
        )
        fig_fi.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    
    # =========================================================================
    # EDA EXPANDER
    # =========================================================================
    with st.expander(" Dataset EDA Details", expanded=False):
        st.markdown('<div class="subtitle-glow">Experience Level Distribution</div>', unsafe_allow_html=True)
        exp_dist = eda_stats.get('experience_level_distribution', {})
        if exp_dist:
            exp_df = pd.DataFrame([
                {'Level': k, 'Count': v}
                for k, v in exp_dist.items()
            ])
            fig_exp = px.pie(
                exp_df, values='Count', names='Level',
                title='Experience Level Distribution',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        
        st.markdown('<div class="subtitle-glow">Work Type Distribution</div>', unsafe_allow_html=True)
        wt_dist = eda_stats.get('work_type_distribution', {})
        if wt_dist:
            wt_df = pd.DataFrame([
                {'Work Type': k, 'Count': v}
                for k, v in wt_dist.items()
            ])
            fig_wt = px.bar(
                wt_df, x='Work Type', y='Count',
                title='Work Type Distribution',
                color='Work Type',
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig_wt, use_container_width=True)
        
        st.markdown('<div class="subtitle-glow">Description Text Statistics</div>', unsafe_allow_html=True)
        desc_stats = eda_stats.get('description_stats', {})
        if desc_stats:
            dcol1, dcol2, dcol3 = st.columns(3)
            dcol1.metric("Non-null Descriptions", f"{desc_stats.get('non_null_count', 0):,}")
            dcol2.metric("Avg Length (chars)", f"{desc_stats.get('avg_length', 0):,.0f}")
            dcol3.metric("Median Length (chars)", f"{desc_stats.get('median_length', 0):,.0f}")
        
        st.markdown('<div class="subtitle-glow">Missing Values Summary</div>', unsafe_allow_html=True)
        missing_vals = eda_stats.get('missing_value_summary', {})
        if missing_vals:
            mv_df = pd.DataFrame([
                {'Column': col, 'Missing Count': info['count'], 'Missing %': info['percentage']}
                for col, info in missing_vals.items()
            ]).sort_values('Missing %', ascending=False)
            st.dataframe(mv_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
