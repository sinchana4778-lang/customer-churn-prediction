# 🚀 Customer Churn Prediction Model

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses.

This project builds an **end-to-end Machine Learning system** that predicts whether a customer is likely to churn based on behavioral and demographic data.

👉 The model is designed to help businesses:

* Reduce customer loss
* Improve retention strategies
* Increase revenue
* Enable targeted marketing

---

## 🎯 Objective

* Predict customer churn using ML models
* Improve churn detection using **threshold tuning & class balancing**
* Provide a **real-time prediction system using Streamlit**
* Deliver business insights from customer behavior

---

## 🧠 Problem Statement

Companies lose significant revenue when customers leave.

Instead of reacting after churn happens, this system:
👉 **predicts churn in advance** so businesses can take action.

---

## ⚙️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Random Forest Classifier**
* **Matplotlib, Seaborn**
* **Streamlit (Frontend UI)**
* **Joblib (Model Persistence)**

---

## 🏗️ Project Architecture

```
Customer Data
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Encoding & Scaling
   ↓
Model Training (Random Forest)
   ↓
Threshold Optimization
   ↓
Prediction
   ↓
Streamlit Dashboard
```

---

## 📊 Model Performance

| Metric         | Value       |
| -------------- | ----------- |
| Accuracy       | ~76%        |
| Recall (Churn) | **~73%** 🔥 |
| Precision      | ~54%        |

👉 Focus was on **high recall** to detect maximum churn customers.

---

## 🧠 Key ML Techniques Used

* Class Imbalance Handling (`class_weight='balanced'`)
* Threshold Optimization (0.5 → 0.3)
* Feature Scaling (StandardScaler)
* Label Encoding for categorical variables

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── data/
├── notebooks/
├── src/
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── columns.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── images/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/sinchana4778/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train Model

```bash
python main.py
```

---

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 💻 Streamlit App Features

* Interactive UI for customer input
* Real-time churn prediction
* Probability-based output
* Clean and user-friendly interface

---
## 🌐 Live Demo
https://customer-churn-prediction-28mkzaraqe23xpzfjtcqmr.streamlit.app/



## 🔄 Virtual Simulation

This project simulates real-world customer behavior:

* Low tenure → high churn
* High monthly charges → high churn
* Month-to-month contracts → high churn

The model learns these patterns to make predictions.

---

## 📈 Business Impact

This system helps companies:

* Identify at-risk customers early
* Launch targeted retention campaigns
* Reduce churn rate
* Increase customer lifetime value

---

## 🚀 Future Improvements

* Deploy using **Streamlit Cloud / Render**
* Add **SHAP Explainability**
* Build **FastAPI backend**
* Integrate with real-time data pipelines

---

## 🙋‍♀️ Author

**Sinchana Gowda**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!

---
