import pandas as pd
import json
import joblib
from pathlib import Path
from src.skill_taxonomy import TECH_SKILLS
from src.data_loader import clean_and_prepare_postings, load_postings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=== 1. DATASET ===")
raw_df = pd.read_csv("data/ai_jobs_global.csv", low_memory=False)
print(f"Raw rows: {len(raw_df)}")
print(f"Raw cols: {len(raw_df.columns)}")

loaded_df = load_postings()
clean_df = clean_and_prepare_postings(loaded_df)
print(f"Clean rows: {len(clean_df)}")
print(f"Kept ratio: {len(clean_df)/len(raw_df)*100:.2f}%")

print("\n=== 2. TAXONOMY ===")
print(f"Taxonomy size: {len(TECH_SKILLS)}")

print("\n=== 3. MODEL EVAL ===")
try:
    metrics = json.loads(Path("artifacts/model_metrics.json").read_text())
    print("Metrics JSON:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
except Exception as e:
    print("Error loading metrics:", e)

try:
    feat_imp = pd.read_csv("artifacts/feature_importance.csv")
    print("\nTop 10 features:")
    print(feat_imp.head(10))
except Exception as e:
    print("Error loading feature imp:", e)

try:
    tech_eda = json.loads(Path("artifacts/tech_eda_stats.json").read_text())
    print(f"\nTech EDA total posts: {tech_eda.get('total_postings')}")
    print("Tech Roles distribution:")
    print(tech_eda.get('role_counts'))
except: pass

print("\n=== 4. TESTS ===")
