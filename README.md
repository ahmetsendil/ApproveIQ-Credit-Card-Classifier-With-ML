# 💳ApproveIQ: Credit Card Classifier With ML 
**Credit Card Applications Analysis Project**
# 🏦 Credit Card Bank Project

This project focuses on predicting whether a credit card application will be approved based on **a set of customer-related features.** It is part of a broader effort to understand and improve **credit risk assessment** using machine learning techniques.

## 📊 Dataset

The dataset used in this project is from Kaggle's **Credit Card Approval Prediction** competition. It includes both categorical and numerical variables that describe applicants' financial and personal status.

- Features include: `Age`, `Income`, `Employment`, `Education`, `Credit Score`, etc.
- Target variable: `TARGET` (1 if approved, 0 rejected)

## 🔍 Objectives

- Perform **Exploratory Data Analysis (EDA)** to understand feature distributions and detect missing or outlier values.
- Apply **data preprocessing** including encoding categorical variables, handling missing values, and scaling.
- Build and compare **multiple classification models** to predict credit card approval.
- Evaluate model performance using metrics such as **accuracy, precision, recall, F1-score**, and **ROC-AUC**.

## 🧹 Data Preprocessing

Key steps:
- Missing values handled with imputation strategies (mean/mode).
- Outliers detected and addressed using IQR method.
- Categorical features encoded using **Label Encoding** and **Ordinal Encoding**.
- Features scaled with **RoboustScaler** to improve model performance.

## 🧠 Models Used

Several machine learning algorithms were applied and compared:
- Random Forest
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
- HistGradient Boosting Classifier

Hyperparameter tuning was performed using **Bayesian Optimization (BayesSearchCV)** and **Optuna** to optimize model performance.

## 🏁 Results

- The best-performing model was **[LightGBM,XGBoost]**, achieving:

**XGBoost Classifier**:
  
Training Time: 1.6224 seconds

Prediction Time: 0.0717 seconds

Accuracy: 1.0

Recall: 1.0

Precision: 1.0

F1 Score: 1.0

AUC: 1.0

**LightGBM Classifier**:

Training Time: 2.65 seconds

Prediction Time: 0.3193 seconds

Accuracy: 1.0

Recall: 1.0

Precision: 1.0

F1 Score: 1.0

AUC: 1.0

Model performance was visualized with confusion matrices and ROC curves.

## 📌 Key Takeaways

- Feature importance analysis revealed that **[STATUS]** played the most significant role in credit approval decisions.
- Best models are  **Decision Tree Classifier**, **XGBoost**,  **LightGBM**, and **HistGradientBoostingClassifier** performed significantly better than simple classifiers.
- Proper preprocessing and tuning can substantially improve model accuracy.

## 📚 Future Work

- Try advanced ensemble techniques (e.g., stacking, voting).
- **GPU** comparison with other libraries.
- I will try to use **Grid CV** for further works 
- Deploy the model using **Streamlit** for real-time predictions.

## 💡 Author

Developed by [Ahmet Şendil](https://www.kaggle.com/ahmetsendil) (https://www.kaggle.com/code/ahmetsendil/approveiq-credit-card-classifier-with-ml)

---

📌 **Feel free to fork this notebook, explore the dataset, and contribute new ideas!**
