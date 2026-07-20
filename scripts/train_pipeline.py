"""
Full training pipeline.
Runs Phase 2 (EDA), Phase 3 (Skill Extraction), and Phase 4 (Model Training)
against the real dataset and saves all artifacts.

Run this once after downloading the data:
    python scripts/train_pipeline.py
"""
import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import load_postings, get_dataset_info, filter_tech_roles, get_postings_with_experience_level, filter_english_postings
from src.skill_extractor import extract_skills_batch, compute_skill_frequencies, compute_skill_frequencies_by_level, compute_skill_frequencies_by_role
from src.eda import compute_summary_stats
from src.model_trainer import train_and_evaluate

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHARTS_DIR = ARTIFACTS_DIR / "charts"


def main():
    start_time = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PHASE 1: Load Data
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: Loading Dataset")
    print("=" * 70)
    
    df = load_postings()
    info = get_dataset_info(df)
    print(f"Loaded {info['rows']:,} rows x {info['columns']} columns")
    print(f"Memory usage: {info['memory_mb']:.1f} MB")
    
    # NEW: Filter to English only to prevent language leakage in TF-IDF
    df, lang_counts = filter_english_postings(df)
    print(f"Language Filtering: Kept {lang_counts['english']} English postings. Dropped {lang_counts['non_english']} non-English.")
    
    # Update info after filtering
    info = get_dataset_info(df)
    
    # Save dataset info
    with open(ARTIFACTS_DIR / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Dataset info saved to {ARTIFACTS_DIR / 'dataset_info.json'}")
    
    # =========================================================================
    # PHASE 2: EDA
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Exploratory Data Analysis")
    print("=" * 70)
    
    eda_stats = compute_summary_stats(df)
    with open(ARTIFACTS_DIR / "eda_stats.json", 'w') as f:
        json.dump(eda_stats, f, indent=2)
    print(f"EDA stats saved. Key numbers:")
    print(f"  Total postings: {eda_stats['total_postings']:,}")
    print(f"  Unique titles: {eda_stats['unique_titles']:,}")
    print(f"  Unique locations: {eda_stats['unique_locations']:,}")
    
    # Filter to tech roles for focused analysis
    tech_df = filter_tech_roles(df)
    print(f"\nTech-filtered postings: {len(tech_df):,} / {len(df):,} ({len(tech_df)/len(df)*100:.1f}%)")
    
    tech_stats = compute_summary_stats(tech_df)
    with open(ARTIFACTS_DIR / "tech_eda_stats.json", 'w') as f:
        json.dump(tech_stats, f, indent=2)
    
    # =========================================================================
    # PHASE 3: Skill Extraction
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Skill Extraction (running on FULL dataset)")
    print("=" * 70)
    
    print("Extracting skills from all job descriptions...")
    t0 = time.time()
    df = extract_skills_batch(df, text_col='description')
    print(f"Skill extraction completed in {time.time()-t0:.1f}s")
    print(f"Average skills per posting: {df['skill_count'].mean():.1f}")
    print(f"Postings with at least 1 skill: {(df['skill_count'] > 0).sum():,}")
    
    # Overall skill frequencies
    skill_freq = compute_skill_frequencies(df)
    skill_freq.to_csv(ARTIFACTS_DIR / "skill_demand.csv", index=False)
    print(f"\nTop 20 skills overall:")
    print(skill_freq.head(20)[['rank', 'skill', 'count', 'percentage', 'category']].to_string(index=False))
    
    # Skill frequencies by experience level
    df_with_level = df[df['formatted_experience_level'].notna()].copy()
    skill_by_level = compute_skill_frequencies_by_level(df_with_level)
    skill_by_level.to_csv(ARTIFACTS_DIR / "skill_demand_by_level.csv", index=False)
    print(f"\nSkill frequencies by level saved ({len(skill_by_level):,} rows)")
    
    # Skill frequencies by role (tech-filtered)
    tech_df_skills = extract_skills_batch(tech_df, text_col='description')
    skill_by_role = compute_skill_frequencies_by_role(tech_df_skills, top_n_roles=15)
    skill_by_role.to_csv(ARTIFACTS_DIR / "skill_demand_by_role.csv", index=False)
    print(f"Skill frequencies by role saved ({len(skill_by_role):,} rows)")
    
    # Also save tech-filtered skill frequencies
    tech_skill_freq = compute_skill_frequencies(tech_df_skills)
    tech_skill_freq.to_csv(ARTIFACTS_DIR / "tech_skill_demand.csv", index=False)
    print(f"\nTop 20 skills in TECH roles:")
    print(tech_skill_freq.head(20)[['rank', 'skill', 'count', 'percentage', 'category']].to_string(index=False))
    
    # =========================================================================
    # PHASE 4: Model Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Model Training")
    print("=" * 70)
    
    # Use the full dataset with extracted skills for model training
    metrics = train_and_evaluate(df, save_dir=ARTIFACTS_DIR)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}:")
    for f in sorted(ARTIFACTS_DIR.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:40s} {size_kb:>8.1f} KB")
    
    print(f"\nKey Metrics:")
    print(f"  Test Accuracy:  {metrics['test_metrics']['accuracy']:.4f}")
    print(f"  Test F1 (macro): {metrics['test_metrics']['f1_macro']:.4f}")
    if metrics['test_metrics']['roc_auc_macro']:
        print(f"  ROC-AUC (macro): {metrics['test_metrics']['roc_auc_macro']:.4f}")
    print(f"  CV Accuracy:    {metrics['cross_validation']['mean_accuracy']:.4f} ± {metrics['cross_validation']['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()
