# Term Deposit Subscription Prediction

## 📊 **Project Overview**
This project predicts whether a client will subscribe to a term deposit based on marketing campaign data. The goal is to build a machine learning model to help target potential clients effectively, improving the bank's marketing strategy.

## 📥 **Dataset**
The dataset used is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing), containing data from a Portuguese banking institution. It includes 45,211 entries and 17 features, such as:
- **Numerical Features**: `age`, `balance`, `duration`
- **Categorical Features**: `job`, `marital`, `education`, `contact`
- **Target Variable**: `y` (whether the client subscribed to a term deposit, `yes` or `no`)

## 🧹 **Data Preprocessing**
### **Handling Missing Data**
- No missing values were detected in the dataset, meaning we didn’t need to impute or handle missing data.

### **Feature Encoding**
- **One-hot encoding** was applied to categorical variables (e.g., `job`, `marital`, `education`, `contact`) to convert them into numerical format suitable for machine learning models.

### **Feature Scaling**
- We scaled the numerical features (`age`, `balance`, `duration`) using **StandardScaler** to ensure that all features contribute equally to the model.

## ⚖️ **Class Imbalance & SMOTE**
### **Why Rebalance?**
- The dataset was imbalanced, with many more **non-subscribers** than **subscribers** (`y = 'yes'`). This could cause the model to be biased towards predicting the majority class (non-subscribers).
  
### **SMOTE (Synthetic Minority Over-sampling Technique)**
- **SMOTE** was applied to **oversample** the minority class (`y = 'yes'`). This helped balance the dataset and improve the model’s performance in predicting subscribers.

## 🤖 **Modeling**

### **Why XGBoost?**
- **XGBoost** was chosen because it is a powerful, scalable model that works well with structured data, handles imbalanced datasets effectively, and has been proven to provide high accuracy on various classification problems.

### **Other Models Evaluated**
- **Logistic Regression**, **Random Forest**, and **SVM** were also tested, but **XGBoost** outperformed them in terms of **accuracy** and **ROC-AUC**.

### **XGBoost Configuration**
- Key hyperparameters like `n_estimators`, `max_depth`, and `scale_pos_weight` were fine-tuned to optimize performance.

## 📈 **Model Evaluation**
- **XGBoost** achieved:
  - **Accuracy**: 90.6%
  - **ROC-AUC**: 0.93
- The **classification report** showed high precision for **non-subscribers**, but precision for **subscribers** was lower, which can be addressed by fine-tuning thresholds.

### **Feature Importance**
- **Top Features**:
  - **`poutcome_success`**: Previous campaign success was the most important feature in predicting subscription.
  - **`contact_telephone`**: Clients contacted via telephone were more likely to subscribe.
  - **`month_mar` and `month_oct`**: Clients contacted in **March** and **October** had higher subscription rates.

## 🔧 **Tools & Libraries Used**
- **Python**: The core language used for building the model.
- **XGBoost**: The primary model used for classification.
- **SMOTE**: Used for rebalancing the dataset.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning utilities, like scaling and splitting data.
- **Matplotlib**: For visualizing results such as the ROC curve and confusion matrix.
- **Jupyter Notebook**: For development and experimentation.

### 🛠️ **Installation**
```bash
# To install required libraries
pip install xgboost smote pandas scikit-learn matplotlib
