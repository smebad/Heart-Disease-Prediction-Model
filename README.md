# ❤️ Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) 
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-green)](https://en.wikipedia.org/wiki/Supervised_learning) 
[![Logistic Regression](https://img.shields.io/badge/Model-Logistic%20Regression-red)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

Predicting heart disease is crucial in early diagnosis and saving lives. In this project, we leverage **Logistic Regression** to predict heart disease based on patient data from the UCI Heart Disease dataset. This repository contains the code, explanations, and insights derived from analyzing and building the model. 

> **Author**: Syed Muhammad Ebad  
> **Date**: 5-Oct-2024  
> [Visit my Kaggle](https://www.kaggle.com/syedmuhammadebad) | [GitHub Profile](https://github.com/smebad) | [Email me](mailto:mohammadebad1@hotmail.com)

---

## 📁 Dataset

The dataset used is the **[Heart Disease UCI](https://www.kaggle.com/datasets/mragpavank/heart-diseaseuci)** dataset, which contains various medical and demographic attributes that could be potential indicators of heart disease.

- **Rows**: 303
- **Columns**: 14 (13 Features + 1 Target)

---

## 🧠 Project Overview

This project aims to predict whether a patient is likely to have heart disease using features such as age, sex, chest pain type, cholesterol level, and more. We utilize **Logistic Regression** as the machine learning model to classify patients based on these features.

### 🛠️ Skills and Tools Used

- **Data Preprocessing**: Pandas, Numpy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Model**: Logistic Regression
- **Evaluation**: Accuracy Score, Heatmaps, Scatter Plots

---

## 📊 Exploratory Data Analysis (EDA)

Before training the model, an extensive EDA was performed to understand the dataset. Key visualizations include:

- **Target Distribution**: Class imbalance between patients with and without heart disease.
- **Correlation Heatmap**: Identify relationships between features.
- **Feature Statistics**: Analyze central tendencies and variability of key attributes.

### Sample Visualizations:

| Target Distribution              | Correlation Heatmap                        |
|----------------------------------|--------------------------------------------|
| ![Target Distribution](./images/target_distribution.png) | ![Correlation Heatmap](./images/correlation_heatmap.png) |

---

## ⚙️ Model Building

### Logistic Regression
We chose **Logistic Regression** as the initial model to predict the binary outcome (heart disease: yes/no). The dataset was split into training and testing sets using an 80/20 split.

## 🔬 Model Performance:

The model achieved an accuracy of approximately X%. While the accuracy metric is a good starting point, further exploration of additional evaluation metrics like precision, recall, and F1-score would give a clearer picture of model performance, especially considering the class imbalance.

## 📌 Key Observations:
* Chest pain type, exercise-induced angina, and maximum heart rate are strongly correlated with heart disease.
* There is a slight class imbalance between patients with and without heart disease.
* Logistic regression provides a baseline model, but further enhancements (like feature engineering or trying different models) could improve accuracy.

