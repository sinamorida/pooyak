# Complete Streamlit Dashboard for Multi-Year Water Consumption Analysis

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------- Data Processing Functions --------------------

def combine_yearly_datasets(dfs: dict):
    long_dfs = []
    for year, df in dfs.items():
        df = df.copy()
        df['Year'] = year
        df_long = df.melt(id_vars=['Subscription Code', 'Customer Name', 'License Type'],
                          var_name='Month', value_name='Consumption')
        long_dfs.append(df_long)
    return pd.concat(long_dfs, ignore_index=True)

def handle_outliers(df, method='cap', z_thresh=3):
    df = df.copy()
    mean = df['Consumption'].mean()
    std = df['Consumption'].std()
    z_scores = (df['Consumption'] - mean) / std
    if method == 'cap':
        df['Consumption'] = np.where(z_scores > z_thresh, mean + z_thresh * std, df['Consumption'])
        df['Consumption'] = np.where(z_scores < -z_thresh, mean - z_thresh * std, df['Consumption'])
    elif method == 'impute':
        df.loc[z_scores > z_thresh, 'Consumption'] = np.nan
        imputer = KNNImputer()
        df['Consumption'] = imputer.fit_transform(df[['Consumption']])
    elif method == 'flag':
        df['Outlier'] = (z_scores.abs() > z_thresh)
    return df

def detect_anomalies_moving(df, window=3, threshold=2):
    df = df.copy()
    df['Rolling Mean'] = df.groupby('Subscription Code')['Consumption'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['Rolling Std'] = df.groupby('Subscription Code')['Consumption'].transform(lambda x: x.rolling(window, min_periods=1).std())
    df['Anomaly'] = (np.abs(df['Consumption'] - df['Rolling Mean']) > threshold * df['Rolling Std'])
    return df

def engineer_features(df):
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce').dt.month.fillna(0).astype(int)
    df['Log Consumption'] = np.log1p(df['Consumption'])
    df['Monthly Change'] = df.groupby('Subscription Code')['Consumption'].diff()
    df['Is Winter'] = df['Month'].isin([10, 11, 12, 1, 2])
    return df

def cluster_customers(df, n_clusters=3):
    pivot_df = df.pivot_table(index='Subscription Code', columns='Month', values='Consumption', aggfunc='mean').fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(pivot_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X)
    pivot_df['Cluster'] = clusters
    return pivot_df.reset_index()

# -------------------- Visualization Functions --------------------

def plot_consumption_by_license(df):
    return px.box(df, x='License Type', y='Consumption', title='Consumption Distribution by License Type')

def plot_customer_timeseries(df, subscription_code):
    sub_df = df[df['Subscription Code'] == subscription_code]
    return px.line(sub_df, x='Month', y='Consumption', color='Year', title=f'Time Series - {subscription_code}')

def plot_cluster_heatmap(cluster_df):
    return px.imshow(cluster_df.drop(columns=['Subscription Code', 'Cluster']),
                     labels=dict(x="Month", y="Customer", color="Consumption"),
                     title='Average Monthly Consumption by Cluster')

# -------------------- Streamlit Dashboard --------------------
st.set_page_config(layout="wide")
st.title("Water Consumption Dashboard (Multi-Year)")

uploaded_files = st.file_uploader("Upload Excel files for multiple years", accept_multiple_files=True, type=['xlsx'])

dfs = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        year = os.path.splitext(uploaded_file.name)[0]
        df = pd.read_excel(uploaded_file)
        dfs[year] = df

    raw_long_df = combine_yearly_datasets(dfs)

    st.sidebar.header("Preprocessing Options")
    outlier_method = st.sidebar.selectbox("Outlier Handling", ["cap", "impute", "flag"])
    outlier_thresh = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0)

    processed_df = handle_outliers(raw_long_df, method=outlier_method, z_thresh=outlier_thresh)
    processed_df = detect_anomalies_moving(processed_df)
    processed_df = engineer_features(processed_df)

    cluster_result = cluster_customers(processed_df)
    st.sidebar.markdown("---")
    selected_chart = st.sidebar.selectbox("Select Chart to View", [

        "License Type Distribution", "Customer Time Series", "Cluster Heatmap"])

    if selected_chart == "License Type Distribution":
        st.plotly_chart(plot_consumption_by_license(processed_df), use_container_width=True)

    elif selected_chart == "Customer Time Series":
        sub_ids = processed_df['Subscription Code'].unique()
        selected_id = st.sidebar.selectbox("Choose Subscription Code", sub_ids)
        st.plotly_chart(plot_customer_timeseries(processed_df, selected_id), use_container_width=True)

    elif selected_chart == "Cluster Heatmap":
        st.plotly_chart(plot_cluster_heatmap(cluster_result), use_container_width=True)

else:
    st.info("Please upload Excel files containing your water consumption data for multiple years.")
