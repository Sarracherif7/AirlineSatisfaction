## Project Overview

This project develops a machine learning pipeline to predict airline customer satisfaction based on various flight and service attributes. It tackles a classification problem using passenger data to categorize satisfaction levels.

The goal is to create a reliable and reproducible training and deployment pipeline following MLOps best practices, including version control, experiment tracking, and continuous integration and deployment (CI/CD).

---

## Problem Statement

Predict whether an airline customer is satisfied or dissatisfied based on features such as gender, customer type, flight class, flight distance, and service ratings (e.g., inflight wifi, boarding ease, seat comfort).

•⁠  ⁠Problem Type: Binary classification  
•⁠  ⁠Dataset: Publicly available airline passenger satisfaction data (preprocessed)  
•⁠  ⁠Outcome: Improve airline services by understanding key satisfaction drivers.

### Feature Engineering  
- Missing values imputed (e.g., arrival delay filled with median).  
- Categorical variables (Gender, Customer Type, Type of Travel, Class) encoded to numeric.  
- No extensive feature creation was applied; focus was on clean, encoded features.  
- SMOTE oversampling was used to balance classes before training.

### Model Performance Comparison  
- Multiple models tested: Random Forest, Logistic Regression, SVM, Gradient Boosting.  
- Metrics tracked: Accuracy, Precision, Recall, F1-score.  
- Random Forest with SMOTE achieved best accuracy (~96%).

### Visualizations  
- Feature importance charts revealed critical predictors like flight distance and inflight services.  
- Confusion matrices detailed classification errors.  
- ROC curves evaluated model discrimination effectiveness.

These analyses informed model selection and potential avenues for enhancement.

---

## Contributors
Kayla de Silva and Sarra Cherif
