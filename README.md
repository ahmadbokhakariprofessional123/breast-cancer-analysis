# Breast Cancer Classification using Machine Learning

## 🔍 Project Summary
- **Dataset:** UCI Breast Cancer Wisconsin (569 samples, 30 features)
- **Models:** Logistic Regression, SVM, Random Forest, KNN, Decision Tree, Naive Bayes
- **Best Result:** SVM achieved 97% accuracy with strong precision and recall


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
EDA was performed to understand feature distributions, correlations, and class differences.

### Diagnosis Distribution
The dataset contains more benign cases than malignant cases, indicating a mild class imbalance.

![Diagnosis Distribution](image/Bar%20Chart.png)

### Feature Correlation
Strong correlations were observed between size-related features such as **radius**, **perimeter**, and **area**, which informed feature selection and dimensionality reduction.

![Correlation Heatmap](image/heatmap.png)

### Unsupervised Clustering
K-Means clustering (after dimensionality reduction) showed a clear separation between malignant and benign cases, suggesting that the underlying data structure supports effective classification.

![K-Means Clustering](image/K%20means%20Clustering.png)

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

## 📈 Model Evaluation
The **Support Vector Machine (SVM)** achieved the best overall performance with **97% accuracy**, demonstrating strong precision and recall across both benign and malignant cases.

![Confusion Matrix – SVM](image/confusion%20matrix.png)

---

## 🏆 Key Results
- SVM and Logistic Regression achieved the highest accuracy (97%)
- Models demonstrated strong generalisation on unseen test data
- Feature correlations highlighted the importance of size-related attributes
- Clustering analysis confirmed natural separability between classes

---

## ⚠️ Limitations & Future Improvements
**Current limitations:**
- Limited feature engineering
- Partial hyperparameter tuning
- Mild class imbalance
- Cross-validation not applied uniformly across all models

**Potential improvements:**
- Apply feature engineering techniques
- Use GridSearchCV for hyperparameter tuning
- Apply cross-validation to all models
- Explore ensemble and advanced models

---

## 🛠️ Tools & Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## 📘 What I Learned
- How to structure an end-to-end machine learning workflow from EDA to evaluation
- The importance of feature correlation analysis in model performance
- How different classifiers behave on high-dimensional medical data
- How proper evaluation (confusion matrix, precision, recall) matters beyond accuracy


## ▶️ How to Run the Project
```bash
pip install -r requirements.txt
jupyter notebook breast-cancer-analysis.ipynb

