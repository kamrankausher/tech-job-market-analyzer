<div align="center">
  <h1> Tech Job Market Skill-Gap Analyzer</h1>
  <p><b>An End-to-End Data Science & Machine Learning Portfolio Project</b></p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" />
</div>

<br/>

The **Tech Job Market Skill-Gap Analyzer** is an end-to-end data pipeline and interactive web application designed to help tech professionals and students understand exactly what skills are currently in demand. 

Built using a dataset of **5,700+ Global Tech & AI Job Postings from 2026**, this tool extracts required skills using Natural Language Processing (NLP), analyzes market trends, and uses a Machine Learning classification model to predict the expected seniority level (Junior, Mid-level, Senior, Lead, Management) of a given job description or resume.

## ✨ Features & UI Elements

- **Premium Interface:** The dashboard features a stunning, animated glassmorphism UI with gradient titles, hover micro-interactions, and 3D-styled icons.
- **NLP Skill Extraction:** Utilizes a custom taxonomy and Regex pattern matching to extract over 200+ modern technologies across domains like Cloud, ML/AI, Databases, and Frontend/Backend development.
- **Interactive Market Dashboard:** A beautiful Plotly and Streamlit-powered dashboard showcasing the top in-demand skills overall, segmented by experience level, and broken down by specific roles.
- **Personalized Skill Gap Analyzer:** Users can paste their resume or a list of their skills to get a tailored gap analysis against real market demand, revealing exactly which high-value skills they are missing.
- **ML Experience Level Predictor:** A Multinomial Logistic Regression classifier trained on TF-IDF text features and binary skill matrices to predict job seniority with high accuracy.

## 📁 Project Structure

```text
.
├── app.py                      # Main Streamlit app entry point with CSS animations
├── pages/                      # Streamlit multi-page UI
│   ├── 1__Market_Dashboard.py  
│   └── 2__Skill_Gap_Tool.py   
├── data/                       # Raw datasets (Not included in repo to save space)
├── src/                        # Core Python modules
│   ├── data_loader.py          # Data ingestion and cleaning
│   ├── skill_taxonomy.py       # 200+ tech skills definitions
│   ├── skill_extractor.py      # NLP extraction logic
│   ├── eda.py                  # Exploratory data analysis
│   ├── model_trainer.py        # ML training and evaluation
│   └── predictor.py            # Real-time inference
├── scripts/                    
│   └── train_pipeline.py       # End-to-end data processing & model training
├── tests/                      # Pytest suite
├── artifacts/                  # Generated models, metrics, and data summaries
└── requirements.txt            # Python dependencies
```

## 🚀 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "Tech Job Market Skill-Gap Analyzer"
   ```

2. **Download the Data:**
   - Place your dataset (e.g., `ai_jobs_global.csv`) inside the `data/` folder.

3. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the Training Pipeline:**
   This script processes the raw data, extracts skills, trains the model, and generates all required ML artifacts for the Streamlit app.
   ```bash
   python scripts/train_pipeline.py
   ```

5. **Start the Application:**
   ```bash
   streamlit run app.py
   ```

## 🧪 Testing

The project includes a full suite of unit tests to verify data loading, NLP extraction, and ML inference logic. Run the test suite using pytest:
```bash
pytest
```

## 📈 Model Architecture

- **Features:** 
  - Text Features: TF-IDF vectorization on job descriptions (max 5,000 features, ngram_range=(1,2)).
  - Skill Features: Binary presence matrix of 200+ taxonomy skills.
- **Algorithm:** Multinomial Logistic Regression (`solver='lbfgs'`, `class_weight='balanced'`).
- **Target Variable:** `formatted_experience_level` (Junior, Mid-level, Senior, Lead, Management).
- **Evaluation:** 5-fold cross-validation, precision, recall, F1-macro, and ROC-AUC. Feature importance is computed via permutation importance.

---
<div align="center">
  <i>Developed as a Data Science & ML Engineering Portfolio Project.</i>
</div>
