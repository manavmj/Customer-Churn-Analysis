# %% [markdown]
# # Customer Churn Analysis & Insights Dashboard
# 
# **Topic:** Telecom Customer Churn Prediction and Business Recommendations  
# 
# ## 1. Problem Understanding
# **What is customer churn?**  
# Customer churn is the rate at which customers stop doing business with an entity. In telecom, it's often users switching to entirely different providers or canceling their plans.
# 
# **Why it matters:**  
# Acquiring new customers is expensive. Retaining existing customers by understanding their pain points yields much higher ROI. Focusing on churn enables companies to secure stable recurring revenue streams.
# 
# **Objective:**
# To analyze structural, behavioral, and demographic features of customers, calculate key performance metrics, evaluate an ML model to identify top churn drivers, and build a conceptual dashboard outlining exact business interventions.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Set aesthetic visual styles for Matplotlib and Seaborn
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# %% [markdown]
# ## 2. Dataset Loading & Description
# *We are loading our synthetic dataset generated specifically for this project.*
# *Run `data_generator.py` if the `.csv` file is missing.*

# %%
df = pd.read_csv('telecom_churn_data.csv')
print(f"Dataset Shape: {df.shape}")
print(df.head())

# %% [markdown]
# ## 3. Data Preprocessing
# Data cleaning is critical before running an analysis. We must handle missing values and encode categorical columns so they are purely numerical.

# %%
# 1. Check for missing values
print("Missing values before processing:")
print(df.isnull().sum())

# We deliberately generated some missing TotalCharges. Since TotalCharges theoretically equals MonthlyCharges * Tenure, 
# we can impute the missing values logically (or rely on the Median). 
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['Tenure_Months'])

print("\nMissing values after processing:")
print(df.isnull().sum())

# 2. Map binary features to Numerical (0 / 1)
binary_cols = ['PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 3. Categorical Encoding (One-Hot)
# We convert multi-class string variables into distinct binary features.
categorical_features = ['Gender', 'Contract', 'PaymentMethod']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Drop CustomerID as it bears no predictive power
df_encoded.drop('CustomerID', axis=1, inplace=True)

print(df_encoded.head())

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)
# 
# Uncovering the 'Why' behind our target variable. Let's visualize the structural shape of our data.

# %%
# 1. Churn Distribution (Class Imbalance)
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Customer Churn Distribution', fontsize=14)
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')

# Add percentage text
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_y() + p.get_height() + 50
    ax.annotate(percentage, (x, y), size=12)

plt.show()

# INSIGHT: Our dataset represents an imbalanced classification problem, which is extremely typical in real-world churn scenarios.

# %%
# 2. Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
plt.title('Churn Rate by Contract Type')
plt.show()

# INSIGHT: Month-to-month contracts have vastly higher churn volumes compared to 1- or 2-year commitments. 
# Lack of long-term commitment directly translates to volatility in customer retention.

# %%
# 3. Tenure Distribution for Churned vs Retained customers
plt.figure(figsize=(10, 5))
sns.kdeplot(df[df['Churn'] == 0]['Tenure_Months'], label='Retained (0)', fill=True, color='green')
sns.kdeplot(df[df['Churn'] == 1]['Tenure_Months'], label='Churned (1)', fill=True, color='red')
plt.title('Distribution of Tenure by Churn Status')
plt.xlabel('Tenure (Months)')
plt.legend()
plt.show()

# INSIGHT: New customers (low tenure) hold the heaviest risk factor for churn. If a customer survives the critical first year, loyalty cements.

# %%
# 4. Correlation Heatmap
plt.figure(figsize=(12, 8))
# Compute correlation strictly on numerical columns
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, cbar=False)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.show()

# INSIGHT: Churn is moderately positively correlated with SupportCalls, and negatively correlated with Tenure. 

# %% [markdown]
# ## 5. Key Metrics
# Let's frame this analysis in a broader business context by deriving standard strategic performance metrics.

# %%
total_customers = len(df)
churned_customers = df['Churn'].sum()
retained_customers = total_customers - churned_customers

# Metrics Execution
churn_rate = (churned_customers / total_customers) * 100
retention_rate = (retained_customers / total_customers) * 100

# CLV (Customer Lifetime Value) estimation
# Avg Monthly Revenue * Avg Customer Lifespan 
avg_monthly_revenue = df['MonthlyCharges'].mean()
avg_tenure = df['Tenure_Months'].mean()
historical_avg_clv = avg_monthly_revenue * avg_tenure

print("====== BUSINESS KPIs ======")
print(f"Total Customer Base: {total_customers}")
print(f"Churn Rate:          {churn_rate:.2f}%")
print(f"Retention Rate:      {retention_rate:.2f}%")
print(f"Avg Monthly Revenue: ${avg_monthly_revenue:.2f}")
print(f"Historical Avg CLV:  ${historical_avg_clv:.2f}")
print("===========================")

# %% [markdown]
# ## 6. Feature Importance via Machine Learning
# Rather than relying solely on correlation, we want a model to explicitly measure what features drive a customer to churn.

# %%
# Define Targets and Predictors
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Retain 20% of data purely for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit Random Forest (We cap depth to avoid overfitting and keep generalized patterns)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
rf_model.fit(X_train, y_train)

# Quick validation sanity check
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Extract feature importances directly from the tree structure
importances = rf_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Render
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='magma')
plt.title('Top 10 Main Drivers of Customer Churn')
plt.show()

# INSIGHT: Support Calls, Tenure, Month-to-Month contracts, and Monthly charges heavily drive the model's prediction.

# %% [markdown]
# ## 7. Dashboard Creation (Plotly Concept)
# Below is a simulation of interactive visualizations you would deploy inside an enterprise dashboard tool (Tableau / Power BI) or via Streamlit/Dash.

# %%
# Concept 1: Executive KPI Banner Row
fig_kpi = go.Figure()

fig_kpi.add_trace(go.Indicator(
    mode="number",
    value=churn_rate,
    number={'suffix': "%", 'font': {'color': 'red'}},
    title={"text": "Current Churn Rate"},
    domain={'x': [0, 0.33], 'y': [0, 1]}
))

fig_kpi.add_trace(go.Indicator(
    mode="number",
    value=historical_avg_clv,
    number={'prefix': "$", 'font': {'color': 'green'}},
    title={"text": "Historical Avg LTV"},
    domain={'x': [0.34, 0.66], 'y': [0, 1]}
))

fig_kpi.add_trace(go.Indicator(
    mode="number",
    value=df['SupportCalls'].mean(),
    number={'font': {'color': 'orange'}},
    title={"text": "Avg Support Calls per User"},
    domain={'x': [0.67, 1], 'y': [0, 1]}
))

fig_kpi.update_layout(title_text="Executive Summary KPIs")
fig_kpi.show()

# %%
# Concept 2: Interactive Drill-Down Trend Chart
# Examining the danger zone where High Support meets High Price points
fig_scatter = px.box(df, x="SupportCalls", y="MonthlyCharges", color="Churn", 
                     title="Impact of Support Calls and Price Segmentation on Churn",
                     labels={"Churn": "Churn Indicator"})
fig_scatter.show()

# %% [markdown]
# ## 8. Detailed Business Recommendations
# 
# ### 1. **Address Month-To-Month Contract Churn**
# - **Problem:** Short-term contracts act as the largest risk category across all metrics. 
# - **Action:** Introduce an explicit loyalty discount sequence aimed at month-to-month users. Trigger a campaign offering them an auto-upgrade 
# to a 1-year plan at a discounted rate starting 30 days before the typical churn peak interval.
# 
# ### 2. **Proactive Intervention on High Support Calls**
# - **Problem:** Customers generating a high baseline density of support calls exhibit extreme churn vulnerability (likely due to systemic issues or intense frustration).
# - **Action:** Algorithmically flag customer accounts crossing 3+ support tickets within a single month. Bypass standard queues and route them directly to a specialized "Customer Success" squad yielding budget flexibility to offer targeted bill credits.
# 
# ### 3. **The First-Year Survival Bump**
# - **Problem:** The KDE Tenure distribution definitively proves newer users churn massively in early months.
# - **Action:** Implement an assertive 90-day comprehensive onboarding strategy. Frequent check-ins, automated welcome series, hardware setups, and hyper-transparent pricing communications.
# 
# ### 4. **Review High Monthly Charges**
# - **Problem:** Feature importance showcases 'Monthly Charges' prominently. Cost fatigue sets in strongly for users unaligned with their actual needs.
# - **Action:** Pioneer a "Plan right-size" algorithm that periodically emails users notifying them if a slightly cheaper plan fully encompasses their historical data footprint. While counter-intuitive for short-term revenue, customers appreciate integrity—this drastically reduces churn and acts as an immediate loyalty moat preventing migration to cheaper competitors.
