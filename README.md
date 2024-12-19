# Term Deposit Subscription Prediction

## 📊 **Project Overview**
In this project, I aimed to predict whether a client will subscribe to a term deposit based on data from a marketing campaign. The goal was to build a machine learning model that could identify clients who are more likely to subscribe, enabling the bank to focus marketing efforts on the most promising leads.

## 📥 **Dataset**
The dataset I used is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). It contains 45,211 entries and 17 features, representing demographic and marketing campaign data from a Portuguese bank.

### **Features**:
- **Numerical Features**: `age`, `balance`, `duration`
- **Categorical Features**: `job`, `marital`, `education`, `contact`
- **Target Variable**: `y` (whether the client subscribed to a term deposit, binary: `yes` or `no`)

### **Why This Dataset**:
I chose this dataset because it offers a real-world example of a marketing campaign and provides a good mix of categorical and numerical data. The challenge of class imbalance also made it a good opportunity to apply techniques like SMOTE for better model performance.

## 🧹 **Data Preprocessing**
### **Handling Missing Data**
There were no missing values in the dataset, so I didn’t need to perform any imputation or other handling for missing data.

### **Feature Encoding**
I applied **one-hot encoding** to categorical features like `job`, `marital`, `education`, and `contact` to convert them into numerical values that could be used by machine learning algorithms.

### **Feature Scaling**
To ensure that all numerical features contribute equally to the model, I scaled features like `age`, `balance`, and `duration` using **StandardScaler**. This step was important to prevent the model from being biased toward features with larger numerical ranges.

## ⚖️ **Class Imbalance & SMOTE**
### **Why Rebalance?**
The dataset was imbalanced, with a significant number of **non-subscribers** compared to **subscribers** (`y = 'yes'`). If I hadn’t addressed this, the model would have been biased toward predicting the majority class (non-subscribers), leading to poor performance on predicting subscribers.

### **SMOTE (Synthetic Minority Over-sampling Technique)**
To handle the imbalance, I used **SMOTE**, which generates synthetic samples for the minority class (`y = 'yes'`). This technique helped balance the dataset and improve the model's performance in predicting subscribers.

## 🤖 **Modeling**
### **Why XGBoost?**
I chose **XGBoost** as the primary model because it handles large datasets effectively, performs well with both categorical and numerical data, and provides high accuracy. Additionally, XGBoost’s ability to handle imbalanced data through **scale_pos_weight** made it a good fit for this project.

### **Other Models Evaluated**
I also tested **Logistic Regression**, **Random Forest**, and **SVM** as alternative models, but **XGBoost** outperformed them in terms of accuracy and ROC-AUC.

### **XGBoost Configuration**
I fine-tuned key hyperparameters like `n_estimators` and `scale_pos_weight` to optimize performance, particularly in handling the class imbalance.

## 📈 **Model Evaluation**
- **XGBoost** achieved:
  - **Accuracy**: 90.6%
  - **ROC-AUC**: 0.93

The **classification report** showed good precision for **non-subscribers**, but precision for **subscribers** could still be improved. The recall for subscribers was strong at 83%, indicating that the model did well in identifying most of the subscribers.

### **Feature Importance**
- **Key Features**:
  - **`poutcome_success`** (previous campaign success) was the most important feature for predicting subscription.
  - **`contact_telephone`** was the most significant contact method, with telephone contact increasing the likelihood of subscription.
  - **`month_mar` and `month_oct`** showed the highest subscription rates, suggesting that marketing campaigns in **March** and **October** are more effective.

### **Confusion Matrix & ROC Curve**
I visualized the **confusion matrix** and **ROC curve** to better understand the model’s ability to distinguish between subscribers and non-subscribers. The results validated that **XGBoost** was able to correctly classify most non-subscribers and a significant portion of subscribers.

## 🔧 **Tools & Libraries Used**
- **Python**: The main programming language I used for the project.
- **XGBoost**: The primary model used for classification.
- **SMOTE**: Applied for rebalancing the dataset.
- **Pandas**: Used for data manipulation and preprocessing.
- **Scikit-learn**: For machine learning utilities like scaling, splitting data, and evaluating models.
- **Matplotlib**: For visualizing results such as the ROC curve and confusion matrix.
- **Jupyter Notebook**: My development environment for experimenting and building the model.

### 🛠️ **Installation**
To install the required libraries, run the following:
```bash
# To install required libraries
pip install xgboost smote pandas scikit-learn matplotlib
