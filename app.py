import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Telco Customer Churn Dashboard")

DATA_URL = "https://raw.githubusercontent.com/munirahsofeaMRT244075/agile-data-science-pma/refs/heads/main/data./telco.csv"
df = pd.read_csv(DATA_URL)
df['TotalCharges'] = df['TotalCharges'].replace(' ', pd.NA)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df = df.drop_duplicates()

contract_filter = st.selectbox("Filter by Contract Type", ["All"] + df['Contract'].unique().tolist())
filtered_df = df if contract_filter == "All" else df[df['Contract'] == contract_filter]

tenure_range = st.slider("Filter by Tenure (months)", 0, 72, (0, 72))
filtered_df = filtered_df[(filtered_df['tenure'] >= tenure_range[0]) & (filtered_df['tenure'] <= tenure_range[1])]

st.write(f"Showing {len(filtered_df)} customers")

st.subheader("Churn Count")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x='Churn', ax=ax1)
st.pyplot(fig1)

st.subheader("Monthly Charges Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['MonthlyCharges'], bins=30, ax=ax2)
st.pyplot(fig2)

st.subheader("Churn by Contract Type")
fig3, ax3 = plt.subplots()
sns.countplot(data=filtered_df, x='Contract', hue='Churn', ax=ax3)
st.pyplot(fig3)

st.subheader("Predict Churn for a New Customer")
tenure_input = st.number_input("Tenure (months)", 0, 72, 12)
monthly_input = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
st.write("Connect your saved model.pkl here to output a churn probability for these inputs.")
