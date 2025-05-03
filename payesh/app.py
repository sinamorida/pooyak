import streamlit as st
import pandas as pd
from analyzer import InteractiveDataAnalyzer

st.set_page_config(page_title="Water Consumption Analytics Dashboard", layout="wide")

st.title("💧 Water Consumption Analytics Dashboard")

# File upload section (multiple files)
uploaded_files = st.file_uploader("Upload one or more Excel or CSV files", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    st.sidebar.header("Select a dataset to analyze")
    file_names = [file.name for file in uploaded_files]
    selected_file_name = st.sidebar.selectbox("Choose a file", file_names)

    selected_file = next(file for file in uploaded_files if file.name == selected_file_name)

    try:
        if selected_file.name.endswith(".csv"):
            df = pd.read_csv(selected_file)
        else:
            df = pd.read_excel(selected_file)

        st.success(f"Dataset '{selected_file.name}' loaded successfully!")

        analyzer = InteractiveDataAnalyzer(df)

        st.sidebar.header("Configuration")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        selected_numeric_col = st.sidebar.selectbox("Select numeric column", numeric_cols)
        selected_categorical_col = st.sidebar.selectbox("Select categorical column", categorical_cols)

        chart_type = st.sidebar.selectbox(
            "Select chart type to display",
            ["None", "Histogram", "Boxplot", "Line Chart", "Outlier Detection", "Pie Chart", "Bar Chart", "Correlation Heatmap"]
        )

        if chart_type != "None":
            # Numeric chart area
            if chart_type in ["Histogram", "Boxplot", "Line Chart", "Outlier Detection", "Correlation Heatmap"]:
                st.subheader("📈 Numeric Visualizations")
                if chart_type == "Histogram":
                    fig = analyzer.plot_histogram(selected_numeric_col)
                elif chart_type == "Boxplot":
                    fig = analyzer.plot_boxplot(selected_numeric_col)
                elif chart_type == "Line Chart":
                    fig = analyzer.plot_line(x_col=df.index, y_col=selected_numeric_col)
                elif chart_type == "Outlier Detection":
                    threshold = st.sidebar.slider("Z-score threshold", 1.0, 5.0, 3.0)
                    fig = analyzer.plot_outliers_zscore(selected_numeric_col, threshold)
                elif chart_type == "Correlation Heatmap":
                    fig = analyzer.plot_correlation_heatmap()
                st.plotly_chart(fig, use_container_width=True)

            # Categorical chart area
            elif chart_type in ["Pie Chart", "Bar Chart"]:
                st.subheader("📊 Categorical Visualizations")
                if chart_type == "Pie Chart":
                    fig = analyzer.plot_pie_chart(category_col=selected_categorical_col, values_col=selected_numeric_col)
                elif chart_type == "Bar Chart":
                    fig = analyzer.plot_bar_chart(category_col=selected_categorical_col, values_col=selected_numeric_col)
                st.plotly_chart(fig, use_container_width=True)

        # Stats and missing
        st.subheader("📉 Statistical Summary")
        st.json(analyzer.statistical_summary(selected_numeric_col))

        st.subheader("❗ Missing Values")
        st.dataframe(analyzer.show_missing_values())

        # Preprocessing
        st.sidebar.header("Preprocessing")
        scale_method = st.sidebar.selectbox("Scaling method", ["standard", "minmax", "robust"])
        if st.sidebar.button("Scale column"):
            scaled_df = analyzer.scale_column(selected_numeric_col, method=scale_method)
            st.dataframe(scaled_df)

    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.info("Please upload one or more datasets to get started.")
    
    