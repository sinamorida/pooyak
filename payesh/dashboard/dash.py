import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Water Usage Dashboard", layout="wide")

# -------- Load Multiple Files -------- #
@st.cache_data
def load_multiple_files(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        filename = file.name.lower()
        try:
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(file, engine="openpyxl")
            elif filename.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                st.warning(f"Unsupported file format: {filename}")
                continue
            df["__source_file__"] = file.name
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error reading file {file.name}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# -------- Preprocess -------- #
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    df = df[numeric_cols].dropna(how='all')
    return df, numeric_cols


# -------- Outlier Detection -------- #
def detect_outliers(df, method="zscore", threshold=3.0):
    df = df.copy()
    if method == "zscore":
        for col in df.select_dtypes(include='number').columns:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[f"{col}_outlier"] = z > threshold
    return df


# -------- Feature Engineering -------- #
def create_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[f"{col}_diff"] = df[col].diff()
        df[f"{col}_rolling_mean"] = df[col].rolling(window=3).mean()
    return df


# -------- Clustering -------- #
def apply_clustering(df, n_clusters=3):
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').dropna().columns
    X = df[numeric_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init='auto')
    df['cluster'] = km.fit_predict(X_scaled)
    return df


# -------- Plot Selector -------- #
def plot_data(df, plot_type, x=None, y=None, color=None):
    fig = None
    if plot_type == "Histogram":
        fig = px.histogram(df, x=x, color=color, nbins=30)
    elif plot_type == "Boxplot":
        fig = px.box(df, x=color, y=y)
    elif plot_type == "Scatter":
        fig = px.scatter(df, x=x, y=y, color=color)
    elif plot_type == "Line":
        fig = px.line(df, x=x, y=y, color=color)
    elif plot_type == "Bar":
        fig = px.bar(df, x=x, y=y, color=color)
    elif plot_type == "Pie":
        fig = px.pie(df, names=color, values=y)
    return fig


# ========== Streamlit App ========== #
st.title("📊 Interactive Water Usage Analysis Dashboard")

uploaded_files = st.file_uploader("Upload Excel or CSV files", type=["xlsx", "xls", "csv"], accept_multiple_files=True)

if uploaded_files:
    df_raw = load_multiple_files(uploaded_files)
    df_clean, numeric_cols = preprocess_data(df_raw)

    with st.sidebar:
        st.subheader("🔧 Options")
        show_outliers = st.checkbox("Detect Outliers", value=True)
        apply_features = st.checkbox("Add Feature Engineering", value=True)
        apply_cluster = st.checkbox("Cluster Customers", value=False)

    if show_outliers:
        df_clean = detect_outliers(df_clean)

    if apply_features:
        df_clean = create_features(df_clean)

    if apply_cluster:
        n = st.slider("Number of Clusters", 2, 10, 3)
        df_clean = apply_clustering(df_clean, n_clusters=n)

    st.success(f"Data shape: {df_clean.shape}")
    st.dataframe(df_clean.head(20))

    # Visualization Panel
    st.markdown("---")
    st.subheader("📈 Create Visualizations")

    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Scatter", "Line", "Bar", "Pie"])
    x_axis = st.selectbox("X Axis", df_clean.columns)
    y_axis = st.selectbox("Y Axis", df_clean.columns)
    color = st.selectbox("Group/Color By", [None] + df_clean.columns.tolist())

    fig = plot_data(df_clean, plot_type, x=x_axis, y=y_axis, color=color)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload at least one .xlsx or .csv file to begin.")
