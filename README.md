Heart Disease Analysis Project Documentation

1. Introduction

1.1 Project Overview
This project aims to analyze heart disease data using machine learning techniques. The goal is to build predictive models to classify the presence of heart disease based on various medical attributes.

1.2 Objectives
- Preprocess and clean the data.
- Perform exploratory data analysis (EDA) to understand dataset characteristics.
- Train and evaluate multiple machine learning models.
- Identify the best-performing model for heart disease prediction.
 
2. Dataset
   
2.1 Data Source
The dataset is loaded from a CSV file and contains multiple medical attributes relevant to heart disease.
 
2.2 Features
The dataset includes attributes such as:

- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting electrocardiographic results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise-induced angina (exang)
- Oldpeak (ST depression induced by exercise)
- Slope of peak exercise ST segment (slope)
- Number of major vessels (ca)
- Thalassemia (thal)
- Target variable (presence of heart disease)
 
3. Exploratory Data Analysis (EDA)
- Visualizations using Seaborn and Matplotlib to explore feature distributions and relationships.
- Statistical tests (e.g., Chi-square test, Spearman correlation) to identify important features.

4. Data Preprocessing
- Handling Missing Values: Checked and imputed missing values if necessary.
- Feature Scaling: Used Min-Max scaling for numerical attributes.
- Data Splitting: Divided dataset into training and testing sets.
- Handling Class Imbalance: Used RandomUnderSampler to balance the dataset.
 
5. Machine Learning Models
The following models were implemented and evaluated:

Logistic Regression

- Accuracy: 0.84
- AUC-ROC: 0.86

Support Vector Machine (SVM)

- Accuracy: 0.82
- AUC-ROC: 0.85

Decision Tree Classifier

- Accuracy: 0.10
- AUC-ROC: 0.10 (Indicates overfitting or poor performance)

Gradient Boosting Classifier

- Accuracy: 0.85
- AUC-ROC: 0.91 (Best-performing model)

6. Model Evaluation & Selection
The Gradient Boosting Classifier demonstrated the best performance with an AUC-ROC of 0.91 and an accuracy of 0.85, making it the recommended model for heart disease prediction. Logistic Regression also performed well and could be considered for interpretability.

 
7. Conclusion & Future Work
Conclusion: The Gradient Boosting Classifier is the most effective model for predicting heart disease based on the dataset used.

Future Work:

- Further optimize hyperparameters of Gradient Boosting.
- Explore deep learning techniques (e.g., neural networks).
- Apply feature engineering techniques to enhance model performance.


