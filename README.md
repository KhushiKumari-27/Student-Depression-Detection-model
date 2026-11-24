# 🧠 Student Depression Risk Detection Model

An end-to-end supervised Machine Learning classification pipeline built with **Logistic Regression** to identify students at risk of depression based on academic, lifestyle, and demographic indicators.

---

## 📌 Problem Statement
Academic pressure, lifestyle habits, and personal environment heavily impact student mental health, yet early risk factors often go undetected. By analyzing structured survey data, this model provides an automated early-warning screening tool to help educational institutions prioritize counseling support.

## 🛠️ End-to-End ML Workflow

1. **Exploratory Data Analysis (EDA) & Cleaning:** Audited 4,416 student records for missing entries, distributions, and data anomalies.
2. **Feature Encoding:**
   * **One-Hot Encoding:** Applied to nominal categorical features (`Relationship_status`, `Marital_status`, `Living_style`, `friend_circle`, `Health_condition`, `Home_division`) to avoid introducing artificial ordinal rankings.
   * **Manual Mapping:** Applied to binary/structured features (`Gender`, `family_background`).
3. **Leakage-Free Preprocessing:** Performed an 80/20 `train_test_split` (3,532 train / 884 test) *prior* to fitting `StandardScaler` on training features only.
4. **Model Training & Evaluation:** Trained a Scikit-Learn `LogisticRegression` classifier and evaluated performance using Confusion Matrices, Precision, Recall, and F1-Score.

---

## 📊 Model Performance & Evaluation

In mental health screening, **Recall** on the positive class (`Yes`) is the most critical metric because minimizing missed at-risk students (False Negatives) is the top priority.

| Metric | Training Set (3,532) | Testing Set (884) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 97.28% | **97.62%** | Strong generalization with **zero overfitting**. |
| **Recall (`Yes` Class)** | 0.94 | **0.94** | Caught **220 of 234** positive cases in unseen test data. |
| **Precision (`Yes` Class)** | 0.96 | **0.97** | Very low false-positive rate (only 7 false alarms). |
| **F1-Score (`Yes` Class)** | 0.95 | **0.95** | High harmonic balance across imbalanced classes. |

### Test Set Confusion Matrix (`n = 884`)
* **True Negatives:** 643 | **False Positives:** 7
* **False Negatives:** 14 | **True Positives:** 220

---

## 🧰 Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn (`LogisticRegression`, `StandardScaler`, `train_test_split`, `classification_report`)
