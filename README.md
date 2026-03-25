# Home Credit Default Risk Analytics Dashboard

An end-to-end data analytics and machine learning project to predict credit default risk using Python, PyCaret, and Power BI.

# Live Dashboard Link:
https://app.powerbi.com/view?r=eyJrIjoiYWIyM2JiYmMtNTJhMi00NmNkLTlmNDYtNjA5MThhZTAwMmUzIiwidCI6ImU1NDhjMjU2LTRkODMtNDRiMi1iZWM2LTcwZDhhOTFhYzIxZSJ9&pageName=028140bb56801c4ea1c4

## **Project Overview**

This project analyzes Home Credit's default risk data and builds a predictive ML model with an interactive Power BI dashboard.

### **Tech Stack**
- **Python 3.14+**
- **PyCaret 3.3.0** - ML automation
- **Pandas/NumPy** - Data processing
- **Plotly/Seaborn** - Visualization
- **Power BI** - Interactive dashboard
- **Jupyter** - Notebook development

## **Project Structure**

```
Credit Default Risk/
├── data/
│   ├── raw/              # Downloaded raw data from Kaggle
│   └── processed/        # Cleaned & engineered features
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Python scripts & modules
├── models/               # Trained ML models
├── power_bi/             # Power BI dashboard files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## **Setup Instructions**

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
- Go to https://www.kaggle.com/c/home-credit-default-risk
- Download the dataset files
- Place them in `data/raw/`

## **Project Phases**

### Phase 1: EDA & Data Exploration
- Load and explore datasets
- Understand features and target variable
- Handle missing values
- Create summary statistics

### Phase 2: Data Preprocessing
- Feature engineering
- Handle imbalanced classes
- Scale/normalize features
- Create train-test splits

### Phase 3: ML Modeling with PyCaret
- Setup classification task
- Compare multiple algorithms
- Tune best model
- Generate predictions

### Phase 4: Model Evaluation
- Cross-validation analysis
- Feature importance analysis
- SHAP explanations
- Performance metrics

### Phase 5: Power BI Dashboard
- Connect to processed data
- Create interactive visualizations
- Build KPI dashboards
- Export predictions

## **Files Overview**

| File | Purpose |
|------|---------|
| `1_eda.ipynb` | Exploratory Data Analysis |
| `2_preprocessing.ipynb` | Data cleaning & feature engineering |
| `3_modeling.ipynb` | ML model training with PyCaret |
| `4_evaluation.ipynb` | Model evaluation & interpretation |
| `dashboard.pbix` | Power BI dashboard |

## **Key Deliverables**

1. ✅ Clean & processed dataset
2. ✅ Trained ML model
3. ✅ Feature importance analysis
4. ✅ Interactive Power BI dashboard
5. ✅ Model predictions CSV

## **Notes**

- Dataset contains multiple tables (application, bureau, credit_card, etc.)
- Target variable: `TARGET` (0 = no default, 1 = default)
- Class imbalance: ~8% positives
- Multiple feature engineering opportunities

---

**Author**: Sayeed Ahmed
**Last Updated**: March 2026
