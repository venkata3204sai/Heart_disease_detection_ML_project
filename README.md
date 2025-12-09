# Heart Disease Detection — Machine Learning Project
<p align="left">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-Gradient%20Boosting-green.svg" alt="Model Used">
  <img src="https://img.shields.io/github/last-commit/venkata3204sai/Heart_disease_detection_ML_project" alt="Last Commit">
  <img src="https://img.shields.io/github/repo-size/venkata3204sai/Heart_disease_detection_ML_project" alt="Repo Size">
</p>

An end-to-end machine learning system that predicts heart disease risk using clinical patient data.
The project includes data preprocessing, model training with hyperparameter tuning, visualization, and a deployed Streamlit app that performs real-time predictions using a saved model and pipeline.

## Project Overview

This project builds an ML pipeline to predict whether a patient has heart disease based on demographic and clinical features. It uses:
- ColumnTransformer + Pipeline for preprocessing
- GridSearchCV for hyperparameter tuning
- Multiple ML models tested (Gradient Boosting performed best overall)
- PCA-based visualisation of decision boundaries
- Streamlit web app for interactive risk prediction
- Saved artefacts (pipeline.pkl and models.pkl) for deployment

Dataset sourced from Kaggle – Heart Disease Dataset.

## Features
### Data Preprocessing
- One-hot encoding of categorical features
- Standard scaling of numerical features
- ColumnTransformer + Pipeline
- 80/20 train–test split
- Cleaned and validated the dataset

### Model Training
- Trained multiple classifiers: Logistic Regression, Random Forest, Gradient Boosting, KNN, SVC
- Hyperparameter tuning using GridSearchCV (5-fold CV)
- Best performance achieved with Gradient Boosting Classifier
- Metrics used: Accuracy, Confusion Matrix, Classification Report

### Visualization
- PCA-based dimensionality reduction
- 2D decision boundary plot for model interpretability

### Deployment
- Saved artefacts:
  - pipeline.pkl (preprocessing pipeline)
  - models.pkl (best classifier)
- Streamlit app for instant prediction based on user inputs

## Repository Structure
CHANGE
```bash
Heart_disease_detection_ML_project/
│
├── app.py                         # Streamlit UI for prediction
├── preprocessing.py               # Data loading & preprocessing pipeline
├── plot.py                        # PCA-based decision boundary visualisation
├── cvd_prediction.ipynb           # Notebook for training, tuning, metrics, plots
├── models.pkl                     # Final trained ML model
├── pipeline.pkl                   # Preprocessing pipeline
├── heart.csv                      # Dataset
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Installation & Setup
1. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit
   ```
2. Run the Streamlit App
   ```bash
   streamlit run app.py
   ```
   Open in browser:
    http://localhost:8501
3. Train the Model(Optional)
   Open the notebook:
   ```bash
   jupyter notebook cvd_prediction.ipynb
   ```

## How the Model Works
1. Preprocessing (pipeline.pkl)
   - OneHotEncoder for: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
   - StandardScaler for numerical features
   - Pass-through for binary (FastingBS)
2. Model Training

GridSearchCV tuned:
  - penalty: l1, l2, elasticnet, none
  - C values: multiple
  - solvers: liblinear, saga
  - max_iter: range of values
  - scoring: accuracy
Gradient Boosting performed best and was selected for deployment.

3. Saving Artefacts
```bash
pickle.dump(best_model, open("models.pkl", "wb"))
pickle.dump(preprocessing_pipeline, open("pipeline.pkl", "wb"))
```

## Streamlit App Preview
ADD PREVIEW

## License
This project is licensed under the MIT License.
See the LICENSE file for details.
