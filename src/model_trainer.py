"""
Model training module.
Trains a classifier to predict experience level from job description text + skill features.
All metrics are computed from real model execution — nothing is hardcoded.
"""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.inspection import permutation_importance

from src.skill_extractor import fit_tfidf, build_skill_features


ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def prepare_features(
    df: pd.DataFrame,
    tfidf_save_path: Path = None,
    max_tfidf_features: int = 5000,
) -> tuple:
    """
    Prepare feature matrix for model training.
    
    Combines:
    1. TF-IDF features from job description text
    2. Binary skill-presence features from taxonomy extraction
    
    Args:
        df: DataFrame with 'description' and 'extracted_skills' columns.
        tfidf_save_path: Path to save the fitted TF-IDF vectorizer.
        max_tfidf_features: Max TF-IDF vocabulary size.
    
    Returns:
        Tuple of (feature_matrix, feature_names, vectorizer)
    """
    # TF-IDF on descriptions
    tfidf_matrix, vectorizer = fit_tfidf(
        df['description'],
        max_features=max_tfidf_features,
        save_path=tfidf_save_path,
    )
    tfidf_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    
    # Binary skill features
    skill_features = build_skill_features(df)
    skill_matrix = csr_matrix(skill_features.values)
    skill_names = list(skill_features.columns)
    
    # Combine
    combined_matrix = hstack([tfidf_matrix, skill_matrix])
    combined_names = tfidf_names + skill_names
    
    return combined_matrix, combined_names, vectorizer


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = 'formatted_experience_level',
    save_dir: Path = ARTIFACTS_DIR,
) -> dict:
    """
    Train a classifier and evaluate with real metrics.
    
    Pipeline:
    1. Filter to rows with non-null target
    2. Prepare features (TF-IDF + skill features)
    3. Train/test split (80/20, stratified)
    4. Train Logistic Regression
    5. Cross-validate (5-fold)
    6. Evaluate on test set
    7. Compute permutation feature importance
    8. Save model, metrics, and feature importance
    
    Returns:
        Dict with all computed metrics.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to labeled rows
    labeled_df = df[df[target_col].notna()].copy()
    print(f"Labeled rows: {len(labeled_df):,}")
    print(f"Class distribution:\n{labeled_df[target_col].value_counts()}")
    
    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labeled_df[target_col])
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")
    
    # Prepare features
    print("\nPreparing features (TF-IDF + skill features)...")
    X, feature_names, vectorizer = prepare_features(
        labeled_df,
        tfidf_save_path=save_dir / "tfidf_vectorizer.joblib",
        max_tfidf_features=5000,
    )
    print(f"Feature matrix shape: {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]:,}, Test size: {X_test.shape[0]:,}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    
    # Cross-validation on training set
    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test set predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # ROC-AUC (one-vs-rest for multiclass)
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    try:
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc = None
    
    # Per-class metrics
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision_macro:.4f} (macro)")
    print(f"Recall:    {recall_macro:.4f} (macro)")
    print(f"F1 Score:  {f1_macro:.4f} (macro)")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f} (macro, OVR)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Permutation feature importance (on a subsample for speed)
    print("Computing permutation feature importance...")
    n_sample = min(2000, X_test.shape[0])
    sample_idx = np.random.RandomState(42).choice(X_test.shape[0], n_sample, replace=False)
    X_test_sample = X_test[sample_idx]
    y_test_sample = y_test[sample_idx]
    
    # Convert sparse matrix to dense array for permutation_importance
    X_test_sample_dense = X_test_sample.toarray() if hasattr(X_test_sample, 'toarray') else X_test_sample
    perm_imp = permutation_importance(
        model, X_test_sample_dense, y_test_sample,
        n_repeats=10, random_state=42, n_jobs=-1, scoring='accuracy'
    )
    
    # Top 30 features by importance
    imp_mean = perm_imp.importances_mean
    imp_std = perm_imp.importances_std
    top_indices = np.argsort(imp_mean)[::-1][:30]
    
    feature_importance = []
    for idx in top_indices:
        feature_importance.append({
            'feature': feature_names[idx],
            'importance_mean': round(float(imp_mean[idx]), 6),
            'importance_std': round(float(imp_std[idx]), 6),
        })
    
    print("\nTop 15 Features by Permutation Importance:")
    for fi in feature_importance[:15]:
        print(f"  {fi['feature']:40s} {fi['importance_mean']:.6f} ± {fi['importance_std']:.6f}")
    
    # Assemble all metrics
    metrics = {
        "model_type": "LogisticRegression",
        "target_variable": target_col,
        "class_names": class_names,
        "dataset_size": {
            "total_labeled": len(labeled_df),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "cross_validation": {
            "n_folds": 5,
            "mean_accuracy": round(float(cv_scores.mean()), 4),
            "std_accuracy": round(float(cv_scores.std()), 4),
            "fold_scores": [round(float(s), 4) for s in cv_scores],
        },
        "test_metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision_macro": round(float(precision_macro), 4),
            "recall_macro": round(float(recall_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "roc_auc_macro": round(float(roc_auc), 4) if roc_auc is not None else None,
        },
        "per_class_report": {
            k: {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1-score": round(v["f1-score"], 4),
                "support": int(v["support"]),
            }
            for k, v in report.items()
            if k in class_names
        },
        "confusion_matrix": conf_matrix,
    }
    
    # Save everything
    joblib.dump(model, save_dir / "model.joblib")
    joblib.dump(label_encoder, save_dir / "label_encoder.joblib")
    print(f"\nModel saved to {save_dir / 'model.joblib'}")
    
    with open(save_dir / "model_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {save_dir / 'model_metrics.json'}")
    
    fi_df = pd.DataFrame(feature_importance)
    fi_df.to_csv(save_dir / "feature_importance.csv", index=False)
    print(f"Feature importance saved to {save_dir / 'feature_importance.csv'}")
    
    # Save feature names for prediction
    joblib.dump(feature_names, save_dir / "feature_names.joblib")
    
    return metrics
