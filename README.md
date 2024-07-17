# Predictive Maintenance Project

## Overview
Predictive maintenance is a technique that uses data analysis tools and techniques to detect anomalies in operations and possible defects in equipment and processes so they can be fixed before they result in failure. This project demonstrates how to implement a predictive maintenance solution using a combination of machine learning, explainable AI, and web technologies.

## Dataset
https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

## Synthetic Dataset
Since real predictive maintenance datasets are generally difficult to obtain and in particular difficult to publish, we present and provide a synthetic dataset that reflects real predictive maintenance encountered in the industry to the best of our knowledge.
### Addressing the Concern: Different Machines with Different Features
You might wonder, "But in the industry, different machines have different features and working conditions, right?" The synthetic dataset is designed to include a broad range of features and scenarios that are commonly found across different types of machines. While it may not capture the specifics of every individual machine, it can generalize the most important factors that affect maintenance needs.

## Techniques used
- **SMOTE (Synthetic Minority Over-sampling Technique):** Used to balance the dataset by generating synthetic samples for minority classes.
- **Stratified K-Fold Cross-Validation:** Ensures each fold maintains the same proportion of observations for each target class as the complete dataset.
- **CatBoost:** A gradient boosting algorithm used for building the predictive model.
- **LIME (Local Interpretable Model-agnostic Explanations):** Provides explanations for the predictions made by the model.
- **Counterfactuals:** Generates the smallest changes to input instances that result in an opposite outcome, providing insights into model behavior.

## Data Preparation
The dataset is preprocessed, and SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle class imbalance.

## Model Training
The CatBoost model is trained using Stratified K-Fold Cross-Validation to ensure robustness.

## Explainability
### LIME
LIME (Local Interpretable Model-agnostic Explanations) is used to provide explanations for the model's predictions.
### Counterfactual Analysis
Counterfactual examples are generated to understand minimal changes needed for opposite outcomes.


## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Harsha2k3/YT_API_Channel_Comparator.git
    YT_API_Channel_Comparator
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt


## Usage

**Run the main script**:
  ```bash
  python main.py
  ```
