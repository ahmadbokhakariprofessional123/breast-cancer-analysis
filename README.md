# Breast Cancer Classification using Machine Learning

## 📌 Project Overview
This project applies machine learning techniques to classify breast cancer tumors as **Malignant** or **Benign** using diagnostic cell features. Multiple classification models were trained and evaluated to compare performance, with the aim of demonstrating how data analytics and predictive modeling can support **early cancer detection**.

---

## 📊 Dataset
- **Source:** UCI Machine Learning Repository – Breast Cancer Wisconsin Dataset  
- **Samples:** 569  
- **Features:** 30 numerical diagnostic features  
- **Target Variable:** Diagnosis (Malignant / Benign)

Each feature represents characteristics of cell nuclei, such as radius, texture, perimeter, and smoothness.

---

## 🔧 Data Preprocessing
The dataset is relatively clean, but several preprocessing steps were applied:

- Removed non-informative columns (`id`, unnamed column)
- Checked for missing values (none found)
- Reduced dimensionality by removing highly correlated features
- Handled outliers using capping instead of removal
- Encoded target variable (Malignant = 1, Benign = 0)
- Split data into training and testing sets (70% / 30%)

These steps helped reduce noise, avoid overfitting, and improve model stability.

---

## 📈 Exploratory Data Analysis (EDA)
EDA was performed to understand feature distributions, correlations, and class differences:

- Visualized diagnosis distribution using bar and pie charts
- Compared malignant vs benign cases using boxplots and histograms
- Generated correlation heatmaps to identify strongly related features
- Used pairplots to explore feature separability
- Applied K-Means clustering, which showed clear separation between the two classes

These insights guided feature selection and model choice.

---

## 🤖 Models Trained
The following classification models were implemented and evaluated:

| Model | Accuracy |
|------|---------|
| Logistic Regression | 97% |
| Support Vector Machine (SVM) | **97% (Best)** |
| Random Forest | 95% |
| K-Nearest Neighbors (KNN) | 95% (Cross-Validated) |
| Decision Tree | 94% |
| Naive Bayes | 93% |

---

## 🏆 Key Results
- **SVM and Logistic Regression** achieved the highest accuracy (97%)
- Models showed strong precision and recall, indicating reliable classification
- Feature correlations revealed that radius, perimeter, and area were highly influential
- Clustering analysis confirmed natural separation between malignant and benign cases

---

## ⚠️ Limitations & Future Improvements
**Current limitations:**
- Limited feature engineering
- Partial hyperparameter tuning
- Class imbalance may affect some models
- Cross-validation not applied to all classifiers

**Potential improvements:**
- Apply feature engineering techniques
- Use GridSearchCV for hyperparameter tuning
- Apply cross-validation across all models
- Explore ensemble and advanced models

---

## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## ▶️ How to Run the Project
```bash
pip install -r requirements.txt
jupyter notebook breast-cancer-analysis.ipynb
