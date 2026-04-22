# Customer Churn Analysis & Insights Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![DataScience](https://img.shields.io/badge/Data-Science-ff69b4)

## 📖 Problem Understanding

**What is Customer Churn?**
Customer churn occurs when customers or subscribers stop doing business with a company or service. It's often expressed as a percentage of subscribers who discontinue their subscriptions within a given time period.

**Why does Churn matter?**
In industries like telecommunications, SaaS, and banking, acquiring a new customer is often 5 to 25 times more expensive than retaining an existing one. High churn rates severely impact revenue and growth. Identifying *why* customers leave (and which ones are likely to leave) allows businesses to take proactive steps to retain them.

**Project Objective:**
This project aims not only to predict which customers will churn but to extract actionable business insights and recommendations based on EDA and Machine Learning feature importance.

## 📂 Dataset
For this project, we are generating a synthetic **Telecom Customer Churn** dataset.
- **Features**: Includes customer demographic data (Age, Gender), Account information (Tenure, Contract Type, Monthly Charges), and usage behavior (Data usage, Support calls).
- **Target Variable**: `Churn` (Yes/No)

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Generate the Data
Run the data generator to create the synthetic dataset (`telecom_churn_data.csv`):
```bash
python data_generator.py
```

### 3. Run the Analysis
You can run the analysis script using an interactive Python environment (like VSCode with Interactive Window or Jupyter):
```bash
python churn_analysis.py
```
Alternatively, you can convert it to or paste the blocks into a simple Jupyter Notebook (`.ipynb`).

## 📊 Project Scope
1. **Data Preprocessing**: Handling missing values, categorical encoding, and feature scaling.
2. **Exploratory Data Analysis (EDA)**: Deep dive into customer behavior and churn distribution.
3. **Key Metrics**: Calculating Retention Rate, Churn Rate, and basic Customer Lifetime Value (CLV).
4. **Feature Importance**: Using a Random Forest model to understand what drives churn.
5. **Dashboard Creation**: Visualized via interactive Plotly charts simulating a business dashboard.
6. **Business Recommendations**: Data-driven strategies for customer retention.

---
*Created as an end-to-end Data Science portfolio project.*
