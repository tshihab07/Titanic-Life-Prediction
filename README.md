# Titanic Survival Prediction

A comprehensive machine learning project to predict passenger survival on the RMS Titanic disaster. This repository contains a complete workflow for data preprocessing, advanced modeling, hyperparameter tuning, model evaluation, and deployment of a production-ready predictive model.

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Data Flow](#data-flow)
    - [Dataset Description](#dataset-description)
    - [Project Workflow](#project-workflow)
- [Key Features](#key-features)
- [Model Overview](#model-overview)
    - [Result](#results)
    - [Model Performance Report](#best-model-performance-summary-xgboost)
- [File Structure](#file-structure)
- [Modeling Pipeline](#modeling-pipeline)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvement](#future-improvement)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

## Project Overview

This project aims to develop and evaluate machine learning models capable of predicting the survival of passengers aboard the RMS Titanic. By analyzing a dataset of passenger demographics, travel class, and other factors, we build predictive models to understand the key characteristics influencing survival chances. The project emphasizes a production-ready approach, including robust data preprocessing, feature engineering, hyperparameter optimization, comprehensive model evaluation, and a deployable inference pipeline.


## Data Flow

### Dataset Description
| Column Name   | Data Type     | Description                                                                         |
| ------------- | ------------- | ----------------------------------------------------------------------------------- |
| `survived`    | int (0 or 1)  | Survival status (0 = No, 1 = Yes)                                                   |
| `pclass`      | int (1, 2, 3) | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)                                            |
| `sex`         | category      | Gender of the passenger (`male` or `female`)                                        |
| `fare`        | float         | Passenger fare (in British Pounds)                                                  |
| `who`         | category      | Person type (`man`, `woman`, `child`)                                               |
| `adult_male`  | bool          | Whether the passenger is an adult male (`True` or `False`)                          |
| `embark_town` | category      | Name of town where the passenger boarded (`Cherbourg`, `Queenstown`, `Southampton`) |
| `alone`       | bool          | Whether the passenger was alone (no family aboard)                                  |


### Project Workflow

1. **Data Loading**: The cleaned dataset `data/titanic_clean.csv` is loaded into a Pandas DataFrame.
2. **Data Splitting**: The dataset is split into training (70%) and testing (30%) sets using a fixed random state for reproducibility.
3. **Feature Scaling**: Numerical features are standardized using `StandardScaler` for models that are sensitive to feature magnitude (Logistic Regression, SVM). Scaling is applied separately to training and testing sets to prevent data leakage.
4. **Model Input**: Preprocessed training data is used for model training and hyperparameter tuning. Preprocessed testing data is reserved for the final, unbiased evaluation of the selected model.


## Key Features

- **Multiple Advanced Algorithms**: Implementation and comparison of **Logistic Regression**, **Random Forest**, **XGBoost**, and **Support Vector Machine (SVM)**.
- **Automated Hyperparameter Optimization**: Utilizes `scikit-learn`'s `GridSearchCV` with 5-fold cross-validation to systematically find the best hyperparameters for each model, ensuring optimal performance.
- **Comprehensive Model Evaluation**: Models are rigorously evaluated using multiple metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC) on both training and test sets. Cross-validation scores provide robust estimates of generalization performance.
- **Overfitting Analysis**: Explicitly compares training and test performance metrics to identify and quantify potential overfitting for each model.
- **Feature Engineering**: Derives new, potentially more predictive features from the raw data.
- **Model Interpretability**: Analyzes feature importance for tree-based models (Random Forest, XGBoost) to understand which factors are most influential in predictions.
- **Production-Ready Components**:
    - Best model and scaler are saved using `joblib`.
    - A complete prediction pipeline function (`predict_survival`) is provided for easy inference on new data.
    - A sample standalone inference script (`titanic_inference.py`) is generated for deployment.


## Model Overview
- Among the evaluated models, **XGBoost** achieved the highest composite score with strong test accuracy, balanced F1-score, and robust calibration.
- After comprehensive evaluation using cross-validation and test set performance, **XGBoost** emerges as the best-performing model.
- It achieves the highest CV Mean (0.842), indicating robust performance across different data splits, and the highest Test Accuracy (0.8134) and Test AUC (0.8542) on unseen data, demonstrating strong generalization.
- The model shows low overfitting (Overfitting Gap = 0.0903) and a high composite score (0.8536), confirming its superior overall performance.
- While **Random Forest** performs well with a CV Mean of 0.8282, **XGBoost**'s higher test accuracy and AUC make it the optimal choice.
- The **Logistic Regression** and **SVM** models exhibit lower performance and higher overfitting gaps.

Therefore, the **XGBoost** model is selected as the final, production-ready model for predicting Titanic survival.

### Results

Based on comprehensive evaluation using a composite scoring system (30% CV Mean, 15% CV Consistency, 15% Test Accuracy, 15% Test F1, 15% Test AUC, 10% Low Overfitting):

| Model             | Test Accuracy | Test F1-Score | Test AUC | CV Mean |
| :---------------- | :-----------: | :-----------: | :------: | :-----: |
| **XGBoost**       | 0.8134        | 0.7664        | 0.8542   | 0.8427  |
| Random Forest     | 0.8060        | 0.7451        | 0.8646   | 0.8282  |
| SVM               | 0.8022        | 0.7535        | 0.8690   | 0.7914  |
| Logistic Regression | 0.7985      | 0.7300        | 0.8176   | 0.8331  |

**🏆 Best Model: XGBoost** - Selected for its optimal balance of performance, robustness, and generalization.

### Best Model Performance Summary (XGBoost):

| Metric             | Value  |
| ------------------ | ------ |
| CV Mean            | 0.8427 |
| CV Std             | 0.0352 |
| Test Accuracy      | 0.8134 | 
| Test F1-Score      | 0.7664 |
| Test AUC           | 0.8542 |
| Overfitting Gap    | 0.0903 |
| Composite Score    | 0.8536 |
| Accuracy Gap       | 0.0903 |
| Precision Gap      | 0.1293 |
| Recall Gap         | 0.0665 |
| F1-Score Gap       | 0.0948 |
| AUC Gap            | 0.109  |
| Overfitting Status | Low    |


## File Structure
```bash
TitanicSurvivalPrediction/
├── data/                             # Dataset storage
│   └── titanic_clean.csv             # Preprocessed Titanic dataset
│
├── model/                            # Saved models and evaluation results
│   ├── AdvancedModeling/             # Outputs from advanced modeling
│   │   ├── BestModel.pkl             # Serialized best-performing model
│   │   ├── BestModelSummary.csv      # Summary of the best model performance
│   │   ├── ModelComparison.csv       # Comparison of multiple models
│   │   ├── ModelPerformance.csv      # Detailed performance metrics
│   │   ├── ModelRanking.csv          # Ranked list of models
│   │   └── OverfittingAnalysis.csv   # Overfitting evaluation results
│   │
│   └── baselineModeling/             # Outputs from baseline modeling
│       ├── baselineModel.pkl         # Serialized baseline model
│       └── BaselineModelSummary.csv  # Summary of baseline performance
│
├── visualizations/                   # Plots and figures
│   ├── Exploratory Data Analysis/    # Visuals from EDA
│   │   ├── BinaryAnalysis-QuickOverview.png
│   │   ├── BinaryAnalysis-SurvivedByChance.png
│   │   ├── BinaryAnalysis-SurvivedByEmbarkTown.png
│   │   ├── BinaryAnalysis-SurvivedByGender.png
│   │   ├── BinaryAnalysis-SurvivedByPassengerClass.png
│   │   ├── boxplot-ageVariableDistribution.png
│   │   ├── boxplot-ageVariableDistributionAcrossGender.png
│   │   ├── countplot-totalSurvivalCount.png
│   │   ├── distplot-ageVariableDistribution.png
│   │   ├── distplot-fareVariableDistribution.png
│   │   ├── heatmap-correlationBetweenTheVariables.png
│   │   └── histogram-ageVariableDistribution.png
│   │
│   └── model_evaluation/             # Visuals for model evaluation
│       ├── All_Model_Comparison.png
│       ├── BestModel_CalibrationPlot.png
│       ├── BestModel_LearningCurves.png
│       ├── BestModel_SHAPSummary.png
│       ├── confusionMatrix - TestSet.png
│       ├── confusionMatrix - TrainingSet.png
│       ├── confusionMatrix_AdvancedModeling_BestModel_TrainvsTest.png
│       ├── ModelComparison_CrossValidationPerformance.png
│       ├── ModelComparison_FinalRanking.png
│       ├── ModelComparison_OverfittingAnalysis.png
│       ├── ModelComparison_ROC-Curves_HighValue.png
│       ├── ModelComparison_TestF1-Score.png
│       ├── ModelComparison_TrainingvsTestAccuracy.png
│       ├── ROC-Curve_AllModels.png
│       ├── ROC-Curve_LogisticRegression_BaselineModel.png
│       ├── ROC-Curve_RandomForest.png
│       ├── ROC-Curve_SVM.png
│       └── ROC-Curve_XGBoosting.png
│
├── advancedModeling.ipynb            # Notebook for advanced model training & evaluation
├── baselineModeling.ipynb            # Notebook for baseline model development
├── preprocessing.ipynb               # Notebook for data preprocessing
├── README.md                         # Project documentation
├── requirements.txt                  # Project dependencies
└── LICENSE                           # License information
└── titanic_inference.py              # Model Inference
```


## Modeling Pipeline

1. **Data Preparation**: Load, engineer features, split, and scale data.
2. **Iterative Model Training & Tuning**: For each model (Logistic Regression, Random Forest, XGBoost, SVM) -
    - Define a parameter grid.
    - Perform `GridSearchCV` with 5-fold CV to find optimal parameters.
    - Train the final optimized model on the full training set.
3. **Model Evaluation**:
    - Generate predictions (train/test) and predicted probabilities for each optimized model.
    - Calculate performance metrics (Accuracy, Precision, Recall, F1, AUC) for train and test sets.
    - Perform 5-fold cross-validation (`cross_val_score`) on the training set for each optimized model.
    - Analyze overfitting by calculating the gap between train and test metrics.
4. **Model Selection**:
    - Rank all models using a composite score that weights Cross-Validation Mean, CV Consistency (1/CV Std), Test Accuracy, Test F1-Score, Test AUC, and low Overfitting.
    - Select the model with the highest composite score as the best performer.
5. **Finalization**:
    - Conduct error analysis (false positives, false negatives) on the best model using the test set.
    - Save the best model object, the scaler, and the list of feature names.
    - Create and save the `predict_survival` function and a sample `titanic_inference.py` script.


## Dependencies

The project requires the following Python libraries:

```bash
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.1
xgboost==2.1.1
shap==0.46.0
joblib==1.4.2
```

---

## Installation

**Clone the Repository**:

```bash
git clone https://github.com/tshihab07/titanic-life-prediction.git
cd titanic-life-prediction
```

**(Recommended) Create a Virtual Environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
```

**Install Dependencies**:
Create a `requirements.txt` file with the dependencies listed above, then run:
```bash
pip install -r requirements.txt
```

**Prepare Data**: Ensure the `data/titanic_clean.csv` file is placed in the `data/` directory.


## Usage

**Run the Full Analysis**:<br>
Execute the main script to perform data preprocessing, train all models, tune hyperparameters, evaluate performance, select the best model, and generate output files.<br><br>

**Use the Inference Script**:<br>
Modify the sample data in `titanic_inference.py` or create your own script using the `predict_survival` function logic to make predictions on new passenger data.

```bash
python titanic_inference.py
```


## Future Improvement

Potential areas for enhancing the model's performance and robustness:

- **Advanced Hyperparameter Tuning**: Explore more sophisticated techniques like Bayesian Optimization (e.g., Optuna, Hyperopt) for potentially better parameter configurations.
- **Ensemble Methods**: Combine predictions from the top-performing models (e.g., Voting Classifier, Stacking) to potentially improve accuracy and robustness.
- **In-Depth Error Analysis**: Perform a detailed investigation into misclassified instances to understand model weaknesses and guide further feature engineering.
- **Model Interpretability (SHAP)**: Integrate SHAP (Shapley Additive exPlanations) values for more detailed, instance-level explanations of model predictions.
- **Learning Curve Analysis**: Generate learning curves to diagnose if the model would benefit from more training data.


## Contributing

Contributions are welcome! Please feel free to submit a pull request.

- Fork the project.
- Create your feature branch
- Commit changes
- Push
- Open a Pull Request


## Contact

E-mail: tushar.shihab13@gmail.com <br>
More Projects: 👉🏿 [Projects](https://github.com/tshihab07?tab=repositories)<br>
LinkedIn: [Tushar Shihab](https://www.linkedin.com/in/tshihab07/)


## License

This project is licensed under the [MIT License](LICENSE).
