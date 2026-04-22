import pandas as pd
import numpy as np

def generate_telecom_data(num_samples=5000):
    np.random.seed(42)
    
    # 1. Demographics
    customer_id = [f"CUST_{i:05d}" for i in range(1, num_samples + 1)]
    gender = np.random.choice(['Male', 'Female'], num_samples)
    age = np.random.randint(18, 80, num_samples)
    senior_citizen = (age >= 65).astype(int)
    
    # 2. Account Info
    tenure = np.random.randint(1, 73, num_samples) # months
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        num_samples, 
        p=[0.5, 0.3, 0.2]
    )
    paperless_billing = np.random.choice(['Yes', 'No'], num_samples)
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
        num_samples
    )
    
    # 3. Usage and Services
    monthly_charges = np.random.uniform(20, 120, num_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 10, num_samples)
    total_charges = np.maximum(total_charges, monthly_charges) # Ensure logical consistency
    
    num_support_calls = np.random.poisson(1.5, num_samples)
    data_usage_gb = np.random.normal(50, 20, num_samples)
    data_usage_gb = np.maximum(data_usage_gb, 0).round(1)
    
    # 4. Synthesize Churn Probability Base
    # Business Logic:
    # - Month-to-month has higher churn.
    # - More support calls -> higher churn.
    # - High monthly charges -> higher churn.
    # - High tenure -> lower churn.
    
    churn_prob = np.zeros(num_samples)
    
    # Contract effect
    churn_prob += np.where(contract == 'Month-to-month', 0.3, 0)
    churn_prob += np.where(contract == 'One year', 0.1, 0)
    
    # Support calls effect
    churn_prob += (num_support_calls * 0.05)
    
    # Tenure effect (longer tenure -> lower churn)
    churn_prob -= (tenure * 0.005)
    
    # Charges effect
    churn_prob += (monthly_charges - 60) * 0.002
    
    # Add random noise
    churn_prob += np.random.normal(0, 0.1, num_samples)
    
    # Clip probabilities between 0.05 and 0.95
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    
    # Final Churn Classification
    churn = np.random.binomial(1, churn_prob)
    churn_label = np.where(churn == 1, 'Yes', 'No')
    
    # Assemble Dataframe
    df = pd.DataFrame({
        'CustomerID': customer_id,
        'Gender': gender,
        'Age': age,
        'SeniorCitizen': senior_citizen,
        'Tenure_Months': tenure,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'SupportCalls': num_support_calls,
        'DataUsage_GB': data_usage_gb,
        'Churn': churn_label
    })
    
    # Introduce a few realistic missing values intentionally to simulate raw data
    missing_indices = np.random.choice(df.index, size=int(num_samples*0.01), replace=False)
    df.loc[missing_indices, 'TotalCharges'] = np.nan
    
    return df

if __name__ == "__main__":
    print("Generating synthetic telecom churn data...")
    dataset = generate_telecom_data(5000)
    filename = "telecom_churn_data.csv"
    dataset.to_csv(filename, index=False)
    print(f"Dataset securely saved to '{filename}'.")
    print(f"Shape: {dataset.shape}")
    print(dataset.head())
