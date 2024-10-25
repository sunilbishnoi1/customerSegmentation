import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

st.title("Customer Segmentation Using RFM and KMeans")

st.sidebar.title("Options")
st.sidebar.markdown("Use the options below to interact with the data and apply clustering.")

# 1. Data Upload Section
st.subheader("Step 1: Upload Data")

file_type = st.selectbox("Choose file type to upload:", ["CSV", "Excel (.xlsx)"])

if file_type == "CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
elif file_type == "Excel (.xlsx)":
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Check if the file is uploaded
if uploaded_file:
    try:
        if file_type == "CSV":
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', parse_dates=['InvoiceDate'])
        elif file_type == "Excel (.xlsx)":
            df = pd.read_excel(uploaded_file)
        
        st.write("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Step 2: Data Preprocessing
    st.subheader("Step 2: Data Preprocessing")

    # Missing Values
    if st.checkbox("Show missing values summary"):
        st.write(df.isnull().sum())

    # Remove Duplicates
    if st.checkbox("Remove Duplicates"):
        df = df.drop_duplicates()
        st.write("Duplicates removed!")

    # Handle Missing Values
    if st.checkbox("Handle Missing CustomerID"):
        df = df.dropna(subset=['CustomerID'])
        st.write("Missing CustomerIDs handled!")
    
    # Display cleaned data
    st.write("Data after preprocessing:")
    st.dataframe(df.head())
    
    st.subheader("Step 3: Outlier Removal")
    
    def remove_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
        lower_limit = df[column].quantile(lower_percentile)
        upper_limit = df[column].quantile(upper_percentile)
        return df[(df[column] > lower_limit) & (df[column] < upper_limit)]

    if st.checkbox("Remove Outliers from Quantity and UnitPrice"):
        df = remove_outliers(df, 'Quantity')
        df = remove_outliers(df, 'UnitPrice')
        st.write("Outliers removed!")
        st.dataframe(df.head())

    # Step 4: RFM Calculation
    st.subheader("Step 4: RFM Calculation")

    if st.button("Calculate RFM"):
        # Recency (last purchase date for each customer)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        current_date = df['InvoiceDate'].max() + pd.DateOffset(1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (current_date - x.max()).days,
            'InvoiceNo': 'count',
            'UnitPrice': lambda x: (x * df['Quantity']).sum()
        }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'UnitPrice': 'Monetary'})

        # Display RFM table
        st.write("RFM Features:")
        st.dataframe(rfm.head())

        # Step 5: Clustering with KMeans
        st.subheader("Step 5: KMeans Clustering")
        
        # Standardize the RFM Features
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        # Elbow Method to find optimal K
        st.write("Choose number of clusters (K) for KMeans:")
        k = st.slider("Select K", 2, 10, 4)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Display the results
        st.write(f"Cluster assignments with K={k}:")
        st.dataframe(rfm.head())

        # Silhouette Score
        silhouette_avg = silhouette_score(rfm_scaled, rfm['Cluster'])
        st.write(f"Silhouette Score for K={k}: {silhouette_avg}")

        # Cluster Profiling
        st.subheader("Step 6: Cluster Profiling")
        cluster_profile = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']
        }).round(2)
        st.write("Cluster Profiles:")
        st.dataframe(cluster_profile)

        # Step 7: Visualization
        st.subheader("Step 7: Visualization")

        # Recency vs Frequency by Cluster
        fig, ax = plt.subplots()
        sns.scatterplot(x=rfm['Recency'], y=rfm['Frequency'], hue=rfm['Cluster'], palette='Set1', ax=ax)
        plt.title('Recency vs Frequency by Cluster')
        st.pyplot(fig)

        # Recency vs Monetary by Cluster
        fig, ax = plt.subplots()
        sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], palette='Set2', ax=ax)
        plt.title('Recency vs Monetary by Cluster')
        st.pyplot(fig)

# Footer
st.sidebar.markdown("### Project by: Sunil Bishnoi")
