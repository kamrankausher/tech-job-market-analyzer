"""
Skill taxonomy for tech/data/ML/AI job market analysis.
Contains ~200 real tech skills organized by category.
Each skill has regex patterns for reliable extraction from job description text.
"""
import re
from typing import Dict, List, Tuple


# Skill taxonomy: {display_name: category}
SKILL_TAXONOMY: Dict[str, str] = {
    # Programming Languages
    "Python": "Programming Languages",
    "R": "Programming Languages",
    "Java": "Programming Languages",
    "Scala": "Programming Languages",
    "C++": "Programming Languages",
    "C#": "Programming Languages",
    "Go": "Programming Languages",
    "Rust": "Programming Languages",
    "JavaScript": "Programming Languages",
    "TypeScript": "Programming Languages",
    "SQL": "Programming Languages",
    "Ruby": "Programming Languages",
    "PHP": "Programming Languages",
    "Swift": "Programming Languages",
    "Kotlin": "Programming Languages",
    "MATLAB": "Programming Languages",
    "Julia": "Programming Languages",
    "Perl": "Programming Languages",
    "Shell Scripting": "Programming Languages",
    "SAS": "Programming Languages",
    
    # ML/DL Frameworks
    "TensorFlow": "ML/DL Frameworks",
    "PyTorch": "ML/DL Frameworks",
    "Keras": "ML/DL Frameworks",
    "scikit-learn": "ML/DL Frameworks",
    "XGBoost": "ML/DL Frameworks",
    "LightGBM": "ML/DL Frameworks",
    "CatBoost": "ML/DL Frameworks",
    "Hugging Face": "ML/DL Frameworks",
    "OpenCV": "ML/DL Frameworks",
    "spaCy": "ML/DL Frameworks",
    "NLTK": "ML/DL Frameworks",
    "JAX": "ML/DL Frameworks",
    "MLflow": "ML/DL Frameworks",
    "Kubeflow": "ML/DL Frameworks",
    "Ray": "ML/DL Frameworks",
    
    # Data Engineering & Processing
    "Pandas": "Data Engineering",
    "NumPy": "Data Engineering",
    "Apache Spark": "Data Engineering",
    "PySpark": "Data Engineering",
    "Hadoop": "Data Engineering",
    "Hive": "Data Engineering",
    "Kafka": "Data Engineering",
    "Airflow": "Data Engineering",
    "dbt": "Data Engineering",
    "Snowflake": "Data Engineering",
    "Databricks": "Data Engineering",
    "ETL": "Data Engineering",
    "Data Pipeline": "Data Engineering",
    "Data Warehousing": "Data Engineering",
    "Data Modeling": "Data Engineering",
    "Apache Flink": "Data Engineering",
    "Apache Beam": "Data Engineering",
    "Luigi": "Data Engineering",
    "Prefect": "Data Engineering",
    "Dagster": "Data Engineering",
    
    # Cloud Platforms & Services
    "AWS": "Cloud Platforms",
    "Amazon Web Services": "Cloud Platforms",
    "Azure": "Cloud Platforms",
    "Google Cloud": "Cloud Platforms",
    "GCP": "Cloud Platforms",
    "S3": "Cloud Platforms",
    "EC2": "Cloud Platforms",
    "Lambda": "Cloud Platforms",
    "SageMaker": "Cloud Platforms",
    "BigQuery": "Cloud Platforms",
    "Redshift": "Cloud Platforms",
    "EMR": "Cloud Platforms",
    "CloudFormation": "Cloud Platforms",
    "Terraform": "Cloud Platforms",
    
    # Databases
    "PostgreSQL": "Databases",
    "MySQL": "Databases",
    "MongoDB": "Databases",
    "Redis": "Databases",
    "Cassandra": "Databases",
    "Elasticsearch": "Databases",
    "DynamoDB": "Databases",
    "Oracle DB": "Databases",
    "SQL Server": "Databases",
    "Neo4j": "Databases",
    "SQLite": "Databases",
    "MariaDB": "Databases",
    "CouchDB": "Databases",
    "InfluxDB": "Databases",
    "Firebase": "Databases",
    
    # Visualization & BI
    "Tableau": "Visualization & BI",
    "Power BI": "Visualization & BI",
    "Matplotlib": "Visualization & BI",
    "Plotly": "Visualization & BI",
    "Seaborn": "Visualization & BI",
    "D3.js": "Visualization & BI",
    "Grafana": "Visualization & BI",
    "Looker": "Visualization & BI",
    "Qlik": "Visualization & BI",
    "Streamlit": "Visualization & BI",
    "Dash": "Visualization & BI",
    
    # DevOps & MLOps
    "Docker": "DevOps & MLOps",
    "Kubernetes": "DevOps & MLOps",
    "Git": "DevOps & MLOps",
    "GitHub": "DevOps & MLOps",
    "GitLab": "DevOps & MLOps",
    "CI/CD": "DevOps & MLOps",
    "Jenkins": "DevOps & MLOps",
    "Ansible": "DevOps & MLOps",
    "Linux": "DevOps & MLOps",
    "Nginx": "DevOps & MLOps",
    "Prometheus": "DevOps & MLOps",
    "Helm": "DevOps & MLOps",
    "ArgoCD": "DevOps & MLOps",
    
    # Web & API Development
    "REST API": "Web & API",
    "GraphQL": "Web & API",
    "FastAPI": "Web & API",
    "Flask": "Web & API",
    "Django": "Web & API",
    "Node.js": "Web & API",
    "React": "Web & API",
    "Angular": "Web & API",
    "Vue.js": "Web & API",
    "Spring Boot": "Web & API",
    "Express.js": "Web & API",
    "HTML": "Web & API",
    "CSS": "Web & API",
    
    # Data Science & Analytics Concepts
    "Machine Learning": "DS/ML Concepts",
    "Deep Learning": "DS/ML Concepts",
    "Natural Language Processing": "DS/ML Concepts",
    "NLP": "DS/ML Concepts",
    "Computer Vision": "DS/ML Concepts",
    "Reinforcement Learning": "DS/ML Concepts",
    "A/B Testing": "DS/ML Concepts",
    "Statistical Analysis": "DS/ML Concepts",
    "Regression": "DS/ML Concepts",
    "Classification": "DS/ML Concepts",
    "Clustering": "DS/ML Concepts",
    "Time Series": "DS/ML Concepts",
    "Recommendation Systems": "DS/ML Concepts",
    "Feature Engineering": "DS/ML Concepts",
    "Model Deployment": "DS/ML Concepts",
    "Generative AI": "DS/ML Concepts",
    "LLM": "DS/ML Concepts",
    "Large Language Models": "DS/ML Concepts",
    "RAG": "DS/ML Concepts",
    "Prompt Engineering": "DS/ML Concepts",
    "Neural Networks": "DS/ML Concepts",
    "Random Forest": "DS/ML Concepts",
    "Gradient Boosting": "DS/ML Concepts",
    "Ensemble Methods": "DS/ML Concepts",
    "Dimensionality Reduction": "DS/ML Concepts",
    "PCA": "DS/ML Concepts",
    
    # Big Data
    "MapReduce": "Big Data",
    "HDFS": "Big Data",
    "Presto": "Big Data",
    "Apache Druid": "Big Data",
    "Delta Lake": "Big Data",
    "Apache Iceberg": "Big Data",
    "Data Lake": "Big Data",
    
    # Security & Compliance
    "Cybersecurity": "Security",
    "Encryption": "Security",
    "OAuth": "Security",
    "SAML": "Security",
    "SOC 2": "Security",
    "GDPR": "Security",
    "HIPAA": "Security",
    
    # Soft Skills & Methodologies
    "Agile": "Methodologies",
    "Scrum": "Methodologies",
    "Kanban": "Methodologies",
    "Jira": "Methodologies",
    "Confluence": "Methodologies",
    
    # Testing
    "Unit Testing": "Testing",
    "pytest": "Testing",
    "Selenium": "Testing",
    "JUnit": "Testing",
    
    # Mobile Development
    "iOS": "Mobile",
    "Android": "Mobile",
    "React Native": "Mobile",
    "Flutter": "Mobile",
    
    # Data Formats & Protocols
    "JSON": "Data Formats",
    "XML": "Data Formats",
    "Parquet": "Data Formats",
    "Avro": "Data Formats",
    "Protobuf": "Data Formats",
    
    # Misc Tools
    "Excel": "Misc Tools",
    "Jupyter": "Misc Tools",
    "Notebooks": "Misc Tools",
    "Slack API": "Misc Tools",
    "Salesforce": "Misc Tools",
    "SAP": "Misc Tools",
}


# Regex patterns for each skill — handles multi-word skills, abbreviations, case variations
# Pattern format: (display_name, compiled_regex)
def _build_skill_patterns() -> List[Tuple[str, re.Pattern]]:
    """
    Build compiled regex patterns for each skill.
    Handles edge cases like:
    - 'R' matching only as standalone word (not inside 'React')
    - 'Go' matching only as standalone (not inside 'Google')
    - Multi-word skills like 'machine learning'
    - Abbreviations like 'NLP', 'LLM'
    """
    patterns = []
    
    # Special patterns for ambiguous short names
    special_patterns = {
        "R": r'\b[Rr]\b(?:\s+(?:programming|language|studio|shiny|package|tidyverse|ggplot)|\s*[,;/])',
        "Go": r'\b(?:Go(?:lang)|Golang)\b',
        "C++": r'(?<!\w)C\+\+(?!\w)',
        "C#": r'(?<!\w)C\s*#(?!\w)',
        "SAS": r'\bSAS\b',
        "D3.js": r'\bD3(?:\.js)?\b',
        "Node.js": r'\bNode(?:\.js|js)?\b',
        "Vue.js": r'\bVue(?:\.js|js)?\b',
        "Express.js": r'\bExpress(?:\.js|js)?\b',
        "CI/CD": r'\bCI\s*/\s*CD\b',
        "REST API": r'\bREST(?:ful)?\s*API\b',
        "A/B Testing": r'\bA\s*/?\s*B\s+test(?:ing|s)?\b',
        "scikit-learn": r'\bscikit[\s-]*learn\b|\bsklearn\b',
        "dbt": r'\bdbt\b',
        "ETL": r'\bETL\b',
        "PCA": r'\bPCA\b',
        "NLP": r'\bNLP\b',
        "LLM": r'\bLLM[s]?\b',
        "RAG": r'\bRAG\b',
        "GCP": r'\bGCP\b',
        "EMR": r'\bEMR\b',
        "S3": r'\bS3\b',
        "EC2": r'\bEC2\b',
        "SOC 2": r'\bSOC\s*2\b',
        "iOS": r'\biOS\b',
        "SQL": r'\bSQL\b',
        "HTML": r'\bHTML\b',
        "CSS": r'\bCSS\b',
        "XML": r'\bXML\b',
        "JSON": r'\bJSON\b',
        "HDFS": r'\bHDFS\b',
        "GDPR": r'\bGDPR\b',
        "HIPAA": r'\bHIPAA\b',
        "SAML": r'\bSAML\b',
        "OAuth": r'\bOAuth\b',
        "Shell Scripting": r'\b(?:shell\s+script(?:ing)?|bash|zsh|powershell)\b',
        "Apache Spark": r'\b(?:Apache\s+)?Spark\b',
        "PySpark": r'\bPySpark\b',
        "Apache Flink": r'\b(?:Apache\s+)?Flink\b',
        "Apache Beam": r'\b(?:Apache\s+)?Beam\b',
        "Apache Druid": r'\b(?:Apache\s+)?Druid\b',
        "Apache Iceberg": r'\b(?:Apache\s+)?Iceberg\b',
        "Delta Lake": r'\bDelta\s*Lake\b',
        "Data Lake": r'\bData\s*Lake\b',
        "Data Pipeline": r'\bData\s+Pipeline[s]?\b',
        "Data Warehousing": r'\bData\s+Warehous(?:e|ing)\b',
        "Data Modeling": r'\bData\s+Model(?:ing|s)\b',
        "Spring Boot": r'\bSpring\s*Boot\b',
        "React Native": r'\bReact\s+Native\b',
        "Amazon Web Services": r'\bAmazon\s+Web\s+Services\b',
        "Google Cloud": r'\bGoogle\s+Cloud\b',
        "Power BI": r'\bPower\s*BI\b',
        "Oracle DB": r'\bOracle(?:\s+(?:DB|Database))?\b',
        "SQL Server": r'\bSQL\s+Server\b',
        "Hugging Face": r'\bHugging\s*Face\b',
        "Machine Learning": r'\b(?:Machine\s+Learning|ML)\b',
        "Deep Learning": r'\bDeep\s+Learning\b',
        "Natural Language Processing": r'\bNatural\s+Language\s+Processing\b',
        "Computer Vision": r'\bComputer\s+Vision\b',
        "Reinforcement Learning": r'\bReinforcement\s+Learning\b',
        "Statistical Analysis": r'\bStatistical\s+Analysis\b',
        "Time Series": r'\bTime\s+Series\b',
        "Recommendation Systems": r'\bRecommend(?:ation|er)\s+System[s]?\b',
        "Feature Engineering": r'\bFeature\s+Engineering\b',
        "Model Deployment": r'\bModel\s+Deployment\b',
        "Generative AI": r'\bGenerative\s+AI\b',
        "Large Language Models": r'\bLarge\s+Language\s+Model[s]?\b',
        "Prompt Engineering": r'\bPrompt\s+Engineering\b',
        "Neural Networks": r'\bNeural\s+Network[s]?\b',
        "Random Forest": r'\bRandom\s+Forest\b',
        "Gradient Boosting": r'\bGradient\s+Boost(?:ing|ed)?\b',
        "Ensemble Methods": r'\bEnsemble\s+Method[s]?\b',
        "Dimensionality Reduction": r'\bDimensionality\s+Reduction\b',
        "Unit Testing": r'\bUnit\s+Test(?:ing|s)?\b',
        "Slack API": r'\bSlack(?:\s+API)?\b',
        "Gradient Boosting": r'\bGradient\s+Boost(?:ing|ed)?\b',
    }
    
    for skill_name in SKILL_TAXONOMY:
        if skill_name in special_patterns:
            pattern = re.compile(special_patterns[skill_name], re.IGNORECASE)
        else:
            # Default: word boundary match, case insensitive
            escaped = re.escape(skill_name)
            pattern = re.compile(rf'\b{escaped}\b', re.IGNORECASE)
        patterns.append((skill_name, pattern))
    
    return patterns


# Pre-compiled patterns (built once at import time)
SKILL_PATTERNS: List[Tuple[str, re.Pattern]] = _build_skill_patterns()


def get_skill_taxonomy() -> Dict[str, str]:
    """Return the full skill taxonomy dict: {skill_name: category}."""
    return SKILL_TAXONOMY.copy()


def get_skill_categories() -> Dict[str, List[str]]:
    """Return skills grouped by category: {category: [skill_names]}."""
    categories = {}
    for skill, cat in SKILL_TAXONOMY.items():
        categories.setdefault(cat, []).append(skill)
    return categories


def get_all_skill_names() -> List[str]:
    """Return all skill names as a sorted list."""
    return sorted(SKILL_TAXONOMY.keys())


def get_skill_count() -> int:
    """Return total number of skills in the taxonomy."""
    return len(SKILL_TAXONOMY)
