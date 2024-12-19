# Predicting Client Subscription to Term Deposit - XGBoost Model

## Overview
This project predicts whether a client will subscribe to a term deposit based on marketing campaign data. The dataset includes client demographic details, previous campaign outcomes, and the contact method used. By building a machine learning model, we can identify clients who are more likely to subscribe, enabling more targeted and efficient marketing efforts.

## Dataset
The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). It contains information collected from a Portuguese banking institution’s direct marketing campaigns. The dataset has 45,211 entries with 17 features.

### **Features**:
- **Numerical Features**: `age`, `balance`, `duration`, `campaign`, `pdays`, `previous`
- **Categorical Features**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- **Target Variable**: `y` (whether the client subscribed to a term deposit, binary: `yes` or `no`)

### **Why This Dataset**:
- The dataset represents real-world marketing data, which makes it a good fit for predicting outcomes of future marketing campaigns.
- The goal of this project is to explore how machine learning can improve decision-making in business by targeting the right clients based on past campaign data.

## Data Preprocessing
To make the dataset suitable for machine learning models, several preprocessing steps were applied:

1. **Data Cleaning**:
   - The dataset was loaded and checked for missing values. No missing values were found, making it ready for model training.
   - Categorical features (e.g., `job`, `marital`, `education`, `contact`) were one-hot encoded to convert them into numerical format.
  
2. **Feature Scaling**:
   - Numerical features (e.g., `age`, `balance`, `duration`) were scaled using **StandardScaler** to ensure that each feature had the same scale. This is important for models like **Logistic Regression** and **SVM**, which are sensitive to feature scales.

3. **Class Imbalance**:
   - The dataset was **imbalanced**, with many more **non-subscribers** (`y = 'no'`) than **subscribers** (`y = 'yes'`).
   - To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was used to generate synthetic examples of the minority class (subscribers), making the dataset more balanced and improving the model’s ability to predict the minority class.

## Modeling
### **XGBoost** was chosen as the primary model for this project. It was selected due to its ability to handle large datasets and perform well with a variety of feature types (both categorical and numerical). 

1. **Model Choice**:
   - XGBoost was used due to its **high performance** on structured/tabular data and its ability to handle **imbalanced datasets** effectively.
   - Other models tested included **Logistic Regression**, **Random Forest**, and **SVM**, but **XGBoost** performed the best in terms of accuracy and **ROC-AUC**.

2. **Hyperparameter Tuning**:
   - The key hyperparameters like **`n_estimators`** and **`scale_pos_weight`** were tuned to handle class imbalance and improve model performance.

3. **Model Evaluation**:
   - The model was evaluated using **accuracy**, **ROC-AUC score**, **classification report**, and **confusion matrix** to assess its ability to predict both classes correctly.

## Model Evaluation
- **XGBoost Model**:
  - **Accuracy**: 90.6%
  - **ROC-AUC**: 0.93
  - The **classification report** showed good precision for non-subscribers (`False` class) but highlighted room for improvement in **precision for subscribers** (`True` class). However, recall for subscribers was strong at 83%, which is a positive result for imbalanced data.

### **Feature Importance**:
- **Key Features**:
  - **`poutcome_success`** (previous campaign success) was the most important feature for predicting subscription.
  - **Contact method** (`contact_unknown`, `contact_telephone`) and the **month of contact** (March, October) were also strong indicators.

### **Confusion Matrix & ROC Curve**:
- The **confusion matrix** and **ROC curve** validated that the model is effective at distinguishing between subscribers and non-subscribers, with a **high true positive rate** for subscribers.

## Business Insights
1. **Key Predictors**:
   - **Poutcome (Previous Campaign Outcome)**: Clients who were successful in previous campaigns are more likely to subscribe to a term deposit.
   - **Contact Method**: **Telephone** contact was found to be a strong predictor for subscription likelihood. Clients contacted via phone were more likely to subscribe compared to other methods.
   - **Seasonality**: **March** and **October** were the months with the highest success rate for subscription.

2. **Marketing Recommendations**:
   - **Target clients** who had a **successful outcome in previous campaigns** (`poutcome_success`).
   - **Focus on telephone-based marketing** as it leads to higher conversion rates compared to other methods.
   - **Optimize marketing campaigns** in **March** and **October** to maximize subscription likelihood.

## Future Work
1. **Hyperparameter Tuning**: Although basic hyperparameter tuning was done, further adjustments can be made to improve precision, especially for the minority class (subscribers).
2. **Model Comparison**: Exploring alternative models like **LightGBM** and **CatBoost** could provide faster training times and potentially better performance.
3. **Ensemble Methods**: Combining different models using **stacking** or **bagging** techniques could further improve accuracy.
4. **Threshold Tuning**: Adjusting the classification threshold could help balance **precision** and **recall** for the minority class (subscribers).

## Conclusion
This project demonstrates how **XGBoost** can be used to predict client subscription to a term deposit based on marketing campaign data. By addressing class imbalance and extracting meaningful features, the model can provide actionable insights for marketing strategies. The **XGBoost** model performed well, achieving **ROC-AUC of 0.93**, and it is ready for further refinement and deployment in real-world marketing campaigns.

## GitHub Repository Contents
- **Data Preprocessing**: Code for cleaning, encoding, and rescaling the data.
- **Model Training**: Code for training **XGBoost**, **Logistic Regression**, **Random Forest**, and **SVM**.
- **Model Evaluation**: Code for evaluating models using **accuracy**, **classification report**, **confusion matrix**, and **ROC-AUC**.
- **Feature Importance**: Code for visualizing the most important features for predicting subscriptions.
- **Resampling with SMOTE**: Code for addressing class imbalance using **SMOTE**.

---
