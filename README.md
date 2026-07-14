# Online Games Popularity Prediction

An end-to-end Machine Learning project for predicting the popularity of Steam games using structured game metadata, community statistics, and Natural Language Processing (NLP).

This project was developed as part of a Machine Learning course and achieved **3rd Place** in a Machine Learning Competition.

---

## Overview

The project is divided into two milestones:

### Milestone 1 — Regression
Predicting the exact number of Steam game recommendations (`RecommendationCount`).

### Milestone 2 — Classification
Classifying games into three popularity categories:
- Low
- Medium
- High

The project covers the complete Machine Learning pipeline from preprocessing to deployment.

---

## Features

- Comprehensive data preprocessing
- Feature engineering
- Outlier detection and handling
- Feature selection
- Natural Language Processing (NLP)
- Model training and evaluation
- Hyperparameter tuning
- Model deployment with Streamlit

---

## Dataset

- **Platform:** Steam
- **Games:** 11,000+
- **Raw Features:** 78
- **Engineered Features:** 81+
- Structured and textual game metadata

---

## Project Pipeline

### Data Preprocessing

- Missing value handling
- Date feature engineering
- IQR-based outlier capping
- Standard Scaling
- Feature engineering
- Feature selection

### NLP Pipeline

- Text cleaning
- TF-IDF Vectorization
- Latent Semantic Analysis (LSA)
- Semantic feature extraction

---

## Machine Learning Models

### Regression

- Decision Tree Regressor
- Random Forest Regressor
- Bagging Regressor

**Best Model**

- **Bagging Regressor**
- **R² Score:** 0.8428
- **RMSE:** 2377.14
- **MAE:** 546.77

---

### Classification

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- CatBoost

| Model | Accuracy | Macro F1 |
|--------|---------:|---------:|
| Logistic Regression | 0.8865 | 0.7224 |
| Random Forest | 0.9121 | 0.7297 |
| SVM | 0.8958 | 0.7277 |
| XGBoost | 0.9081 | 0.7504 |
| LightGBM | 0.9187 | 0.7477 |
| **CatBoost** | **0.9170** | **0.7517** |

---

## Techniques Used

- Feature Engineering
- Feature Selection
- TF-IDF
- Latent Semantic Analysis (LSA)
- SMOTE
- Hyperparameter Tuning
- Ensemble Learning
- Cross Validation
- Model Deployment

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- NLTK
- Streamlit
- Matplotlib
- Joblib

---

## Deployment

The final regression model was deployed using **Streamlit**, enabling users to predict game popularity through an interactive web interface.

### Live Demo

👉 **[Add Streamlit Deployment Link Here]**

---

## Project Structure

```text
├── data/
├── preprocessing/
├── feature_engineering/
├── nlp/
├── regression/
├── classification/
├── deployment/
├── models/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

---

## Team

- Nouran Mahmoud
- Basmala Ezzat
- Rokaya Mohamed
- Sara Emad
- Salma Fawzy
- Fatma Alzahraa

---

## Acknowledgments

Special thanks to **Dr. [Instructor Name]**, our Machine Learning course instructor, for the valuable guidance and insightful feedback throughout this project.

---

## License

This project was developed for educational purposes as part of a Machine Learning course.
