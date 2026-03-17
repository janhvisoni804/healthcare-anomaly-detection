import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Healthcare Anomaly Detector", layout="wide")

st.title("🏥 AI-Driven Healthcare Anomaly Detection System")
st.markdown("This system monitors patient vitals like **Heart Rate**, **SpO2**, and **Body Temperature**.")

# --- MILESTONE 1 & 2: Data Generation ---
def generate_health_data():
    np.random.seed(42)
    rows = 200
    data = {
        'Heart_Rate': np.random.normal(75, 5, rows),
        'SpO2': np.random.normal(98, 1, rows),
        'Body_Temp': np.random.normal(37, 0.5, rows)
    }
    df = pd.DataFrame(data)
    
    anomalies = pd.DataFrame({
        'Heart_Rate': [130, 40, 145], 
        'SpO2': [85, 88, 80], 
        'Body_Temp': [39.5, 35.0, 40.1]
    })
    return pd.concat([df, anomalies], ignore_index=True)

df = generate_health_data()

# --- MILESTONE 3: Model Training ---
model = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_Score'] = model.fit_predict(df[['Heart_Rate', 'SpO2', 'Body_Temp']])
df['Status'] = df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'})

def classify_severity(row):
    if row['Status'] == 'Normal':
        return 'LOW'
    elif row['Heart_Rate'] > 120 or row['SpO2'] < 90:
        return 'HIGH'
    else:
        return 'MEDIUM'

df['Severity'] = df.apply(classify_severity, axis=1)

# --- MILESTONE 5: Dashboard ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Live Patient Vitals Data")
    st.dataframe(df.tail(10))

with col2:
    st.subheader("🚨 Detected Anomalies")
    anomaly_df = df[df['Status'] == 'Anomaly']
    st.write(f"Total Anomalies Found: {len(anomaly_df)}")
    st.dataframe(anomaly_df[['Heart_Rate', 'SpO2', 'Body_Temp', 'Severity']])

st.divider()
st.subheader("📈 Anomaly Visualization")
fig, ax = plt.subplots(figsize=(10, 4))
sns.scatterplot(data=df, x='Heart_Rate', y='SpO2', hue='Severity', 
                palette={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}, ax=ax)
st.pyplot(fig)

if len(df[df['Severity'] == 'HIGH']) > 0:
    st.error("⚠️ CRITICAL ALERT: High-risk patients detected!")

