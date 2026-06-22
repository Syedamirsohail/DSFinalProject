# Breast Cancer Recurrence Prediction using Machine Learning

## 📋 Overview

This project investigates whether **clinical and pathological features** of breast cancer patients — such as tumour size, lymph node involvement, and degree of malignancy — can be used to predict whether a patient will experience **cancer recurrence** after treatment. Early identification of high-risk patients can support closer monitoring and more targeted treatment planning.

A key focus of this project is comparing **Label Encoding vs One-Hot Encoding** for categorical clinical variables, and demonstrating how encoding choice can introduce misleading numerical relationships in medical datasets.

## 🎯 Research Question

Can clinical and pathological features be used to classify whether a breast cancer patient will experience recurrence after treatment, and how does the choice of categorical encoding method affect the reliability of this analysis?

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — Breast Cancer dataset
- **Original size:** 286 rows, 9 categorical features + target (`Class`)
- **After cleaning:** 277 rows (9 rows removed due to missing values, originally marked as `"?"`)
- **Target classes:** `no-recurrence-events` vs `recurrence-events` (imbalanced, approx. 70/30 split)
- **Features:** age, menopause, tumour size, invasive nodes, node caps, degree of malignancy, breast side, breast quadrant, irradiation status

## 🔧 Methodology

1. **Data Cleaning** — Replaced `"?"` placeholders with `NaN` and dropped incomplete rows.
2. **Encoding Comparison** — Applied both Label Encoding and One-Hot Encoding to categorical features, to evaluate their effect on correlation analysis and model performance.
3. **Exploratory Data Analysis** — Correlation heatmaps, class distribution analysis, feature relationship checks.
4. **Class Imbalance Handling** — Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set only (after train-test split, to avoid data leakage).
5. **Model Training** — Logistic Regression, Decision Tree, and Random Forest classifiers.
6. **Hyperparameter Tuning** — `GridSearchCV` with 5-fold cross-validation, optimised for F1 score (not accuracy, due to class imbalance).
7. **Evaluation** — Accuracy, Precision, Recall, F1 Score, and Confusion Matrix, with particular emphasis on Recall given the clinical cost of false negatives.

## 🤖 Models Used

| Model | Why It Was Chosen |
|---|---|
| Logistic Regression | Simple, interpretable baseline |
| Decision Tree | Handles categorical splits naturally, interpretable |
| Random Forest | Ensemble method, provides feature importance, robust to overfitting |

Note: SVM, ANN, and CNN approaches were reviewed in the literature but not implemented — the dataset (277 rows, fully categorical, tabular) is too small for neural network approaches to be reliable, and interpretability was prioritised given the clinical context.

## 📈 Key Results

- **Random Forest** was the best-performing model (~78.6% accuracy, F1 ≈ 0.571), with default hyperparameters already near-optimal — tuning gave no further improvement, suggesting a good bias-variance balance for this dataset size.
- **Logistic Regression** achieved ~76.8% accuracy but only ~31% recall, meaning it missed the majority of actual recurrence cases — a clear example of why accuracy alone is misleading on imbalanced medical data.
- **Decision Tree** performance *worsened* after hyperparameter tuning (F1 dropped from ~0.457 to ~0.345), as overly restrictive parameters limited useful splits.
- **Top predictive features (Random Forest):** tumour size (~24% importance), degree of malignancy, and lymph node involvement (`inv_nodes`, `node_caps`).

## 🔍 Key Finding: The False Ordinal Relationship

Label Encoding assigns categories a number based on **alphabetical order**, not real-world meaning. This produced a misleading **strong negative correlation between age and menopause status** — purely because the "pre-menopausal" category alphabetically received the highest number while post-menopausal categories received lower numbers. One-Hot Encoding resolved this, revealing the expected clinically accurate pattern: older age groups strongly associated with post-menopausal status.

This demonstrates a broader principle: **Label Encoding can create false correlations between categorical variables, because the correlation reflects arbitrary alphabetical numbering of categories rather than their true real-world relationship.**

## ⚠️ Limitations

- Small dataset (277 rows) limits model complexity and generalisability.
- No feature scaling applied prior to Logistic Regression, despite tuning its regularization parameter (`C`).
- Dataset lacks hormone receptor status (ER/PR), a known strong predictor of recurrence in the wider literature.
- Dropping rows with missing values (rather than imputing) reduces sample size and risks bias if missingness is not random.

## 🚀 Future Work

- Apply feature scaling and re-evaluate Logistic Regression performance.
- Explore Explainable AI techniques (SHAP, LIME) for individual prediction interpretability.
- Test cost-sensitive learning as an alternative to SMOTE.
- Validate findings on a larger, more feature-rich clinical dataset.

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn (LabelEncoder, train_test_split, GridSearchCV, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

## 📁 Project Structure

```
├── Breast_Cancer_version9.ipynb   # Main analysis notebook
├── Report_BreastCancer.docx       # Full project report
└── README.md                      # This file
```

## ▶️ How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn`
3. Open and run `Breast_Cancer_version9.ipynb` in Jupyter Notebook or JupyterLab

## 📚 Reference

Dataset: UCI Machine Learning Repository — Breast Cancer Data Set, donated by M. Zwitter and M. Soklic, Institute of Oncology, University Medical Center, Ljubljana.
