# Auto Insurance Fraud Detection using Machine Learning

A data-driven system that classifies auto insurance claims as **fraudulent** or **legitimate**, helping insurers flag suspicious claims early, cut financial losses, and reduce manual investigation effort.

---

## Overview

Fraudulent insurance claims cost the industry billions every year. This project builds an end-to-end supervised machine learning pipeline that learns the patterns separating genuine claims from fraudulent ones, using features like claim amount, policy term, financial hardship, and high-risk indicators.

The workflow is: **clean the data → explore it (EDA) → engineer features → handle class imbalance → train and benchmark four models → evaluate on accuracy, precision, recall, and F1.**

## Dataset

- **Source:** [Fraud Detection Dataset (Zenodo)](https://zenodo.org/record/13381118)
- Structured insurance-claim records with both categorical and numerical features.

| Feature | Description |
|---|---|
| Claim Amount | Total amount claimed (USD) |
| Policy Term | Duration of the policy |
| Claim Type | Vehicle, health, etc. |
| Police Report Filed | Whether a report was submitted |
| Financial Hardship | Claimant's financial situation |
| High-Risk Indicator | Binary high-risk flag |

## Approach

**Data preprocessing**
- Replaced placeholder values (`?`) and imputed missing data — mean for numeric, mode for categorical.
- Encoded categorical variables (One-Hot / Label Encoding) and scaled numeric features.
- Engineered date features (e.g. policy bind year/month) from raw timestamps.

**Exploratory data analysis**
- Univariate and bivariate analysis, correlation heatmaps, and boxplots.
- Found fraudulent claims skew toward **higher claim amounts**, and that **high-risk indicators** and **financial hardship** are among the strongest predictors of fraud.

**Handling class imbalance**
- Fraudulent claims are the minority class, so **SMOTE** was used to rebalance the training data and avoid a model that simply predicts "not fraud" every time.

**Models compared**
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost
- Support Vector Machine (SVM)

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 95% | 97% | 98% | 97% |
| Random Forest | 98% | 100% | 98% | 96% |
| **XGBoost** | **97%** | **99%** | **98%** | **98%** |
| SVM | 97% | 99% | 98% | 98% |

**XGBoost** gave the strongest, most balanced performance, capturing the complex, non-linear relationships in the data better than the simpler models.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `matplotlib` · `seaborn`

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Akhilkarumanchi05/Capstone-Insurance-Fraud-Detection.git
cd Capstone-Insurance-Fraud-Detection

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook
```

Then open the notebook and run the cells top to bottom. Update the dataset path at the top of the notebook to point to your local copy of the data.

## Future Work

- Explore deep learning and ensemble stacking for further gains.
- Add model explainability (SHAP) to show *why* a claim is flagged.
- Deploy as an API for real-time claim scoring.

---

*Built by [Akhil Karumanchi](https://github.com/Akhilkarumanchi05) · [LinkedIn](https://www.linkedin.com/in/akhil-karumanchi-56440922a/)*
