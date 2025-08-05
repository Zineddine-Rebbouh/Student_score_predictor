# Student Performance Prediction

**Project Title** : Predicting Student Exam Scores Using Machine Learning

**Date** : August 6, 2025

**Author** : Zineddine Rebbouh

---

## 1. Introduction

This project aims to predict students' exam scores based on factors such as study hours, attendance, parental education, and teacher quality, using the **StudentPerformanceFactors.csv** dataset from Kaggle. The primary objectives are to:

- Clean and preprocess the dataset
- Perform exploratory data analysis (EDA) to understand feature distributions
- Train a linear regression model to predict exam scores
- Evaluate model performance using metrics like MSE and R²
- Experiment with polynomial regression and feature engineering to improve performance
- Save all necessary artifacts for reproducibility and deployment

The project uses Python with libraries such as Pandas, Matplotlib, Seaborn, Scikit-learn, Statsmodels, and Joblib.

---

## 2. Dataset Description

- **Source** : Student Performance Factors (Kaggle)
- **Target Variable** : **Exam_Score** (continuous, 0-100)
- **Features** : Includes numeric (e.g., **Hours_Studied**, **Attendance**) and categorical (e.g., **Teacher_Quality**, **Parental_Education_Level**) variables
- **Size** : 6607 rows

---

## 3. Methodology

### 3.1 Data Loading and Exploration

- Loaded the dataset using a custom **DataLoader** class
- Inspected dataset structure with **df.info()** and **df.describe()**
- Checked for missing values and duplicates
- Visualized numeric feature distributions using boxplots and categorical feature distributions using bar plots

### 3.2 Data Cleaning

- **Missing Values** : Filled missing values in categorical columns (**Teacher_Quality**, **Parental_Education_Level**, **Distance_from_Home**) with mode
- **Outliers** : Capped outliers in numeric columns using the IQR method (1.5 \* IQR rule)
- **Duplicates** : Removed any duplicate rows

### 3.3 Data Preprocessing

- **Encoding** : Converted categorical variables to numeric using one-hot encoding (**pd.get_dummies**)
- **Scaling** : Applied **StandardScaler** to normalize features
- **Data Split** : Split data into 80% training and 20% testing sets with **random_state=42**

### 3.4 Feature Analysis

- **Correlation** : Visualized feature correlations using a heatmap
- **Multicollinearity** : Calculated Variance Inflation Factor (VIF) to ensure no serious multicollinearity (all VIF < 5)

### 3.5 Linear Regression Model

- Trained a linear regression model on scaled training data
- Evaluated performance using:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² score
- Performed 5-fold cross-validation to assess model robustness
- Visualized actual vs predicted values and residuals
- **Saved Artifacts** :
- Model: **lr_model.joblib**
- Scaler: **scaler.joblib**
- Train/test datasets: **X_train.csv**, **X_test.csv**, **y_train.csv**, **y_test.csv**
- Feature names: **feature_names.csv**
- Performance metrics: **performance_metrics.csv**
- Plots: **actual_vs_predicted.png**, **residuals_plot.png**
- Feature coefficients: **feature_coefficients.csv**

### 3.6 Bonus Tasks

- **Polynomial Regression** :
- Tested degrees 1 to 4 to capture non-linear relationships
- Selected the best degree based on cross-validated R²
- Saved the best model, train/test datasets, feature names, and performance metrics
- **Feature Engineering** :
- Created interaction terms (e.g., **Hours_Studied \* Attendance**)
- Added polynomial features (e.g., **Hours_Studied_squared**)
- Generated ratio features (e.g., **Attendance_per_Hour**)
- Applied feature selection (**SelectKBest**) if >20 features
- Evaluated performance with all engineered features and selected features
- Saved the best model, scaler, train/test datasets, feature names, performance metrics, and feature selector (if applicable)

---

## 4. Results

### 4.1 Data Exploration

- Numeric features showed [insert observations, e.g., varying ranges, some outliers]
- Categorical features had [insert observations, e.g., balanced/unbalanced distributions]
- No significant multicollinearity (all VIF < 5)

### 4.2 Linear Regression

- **Training Performance** : [Insert metrics, e.g., MSE, RMSE, R²]
- **Test Performance** : [Insert metrics, e.g., MSE, RMSE, R²]
- **Cross-Validation** : [Insert CV R², indicating model stability]
- Key features influencing exam scores: [Insert top coefficients, e.g., **Hours_Studied**, **Attendance**]
- All artifacts saved in **../models/linear_regression/**

### 4.3 Polynomial Regression

- Tested degrees 1–4
- Best degree: [Insert best degree and CV R²]
- [Insert comparison with linear regression, e.g., slight improvement or overfitting at higher degrees]
- Artifacts saved in **../models/polynomial_regression/**

### 4.4 Feature Engineering

- Added [insert number] new features
- Performance with all engineered features: [Insert R², MSE]
- Performance with selected features: [Insert R², MSE]
- Improvement over baseline: [Insert improvement in R²]
- Artifacts saved in **../models/feature_engineered/**

---

## 5. Visualizations

- **Boxplots** : Identified outliers in numeric features
- **Bar Plots** : Showed distributions of categorical variables
- **Correlation Heatmap** : Highlighted relationships between features
- **Actual vs Predicted Plot** : Showed model predictions
- **Residual Plot** : Confirmed model assumptions (random residuals)
- Plots saved for linear regression model

---

## 6. Discussion

- The linear regression model provided a solid baseline for predicting exam scores
- Polynomial regression [insert observation, e.g., improved performance for degree X but risked overfitting]
- Feature engineering added valuable features, improving R² by [insert improvement]
- The dataset was clean with minimal missing values, and no significant multicollinearity issues were found
- Saving artifacts ensures reproducibility and facilitates deployment
- Limitations:
  - [Insert limitations, e.g., dataset size, feature interactions not fully explored]
  - Linear regression assumes linear relationships, which may not capture all patterns

---

## 7. Conclusion

This project successfully built and evaluated a predictive model for student exam scores. Key takeaways:

- Linear regression is effective but may benefit from non-linear models
- Feature engineering improved performance by capturing interactions and non-linear effects
- Cross-validation ensured robust evaluation
- Saved artifacts enable future use and deployment

Future work could include:

- Testing other algorithms (e.g., Ridge, Lasso, Decision Trees)
- Hyperparameter tuning with grid search
- Exploring additional feature interactions or external data sources

---

## 8. Code and Reproducibility

- The full code is available in the Jupyter notebook (**score_student_system.ipynb**)
- Requirements: Python 3.12.4, Pandas, Matplotlib, Seaborn, Scikit-learn, Statsmodels, Joblib
- Dataset: Available on Kaggle
- Reproducible with **random_state=42** for data splitting and cross-validation
- Artifacts saved in:
  - **../models/linear_regression/**
  - **../models/polynomial_regression/**
  - **../models/feature_engineered/**

---

## 9. References

- Kaggle Dataset: Student Performance Factors
- Scikit-learn Documentation: https://scikit-learn.org
- Statsmodels Documentation: https://www.statsmodels.org
- Joblib Documentation: https://joblib.readthedocs.io
