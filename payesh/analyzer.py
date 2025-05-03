import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore, skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import streamlit as st

class InteractiveDataAnalyzer:
    def __init__(self, df):
        self.df = df

    def filter_by_column_value(self, column, value):
        return self.df[self.df[column] == value]

    def filter_by_range(self, column, min_val=None, max_val=None):
        df_filtered = self.df
        if min_val is not None:
            df_filtered = df_filtered[df_filtered[column] >= min_val]
        if max_val is not None:
            df_filtered = df_filtered[df_filtered[column] <= max_val]
        return df_filtered

    def filter_by_date_range(self, date_column, start_date, end_date):
        df = self.df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]

    def plot_histogram(self, column, nbins=50):
        fig = px.histogram(self.df, x=column, nbins=nbins, title=f"Histogram of {column}")
        fig.update_layout(bargap=0.1)
        return fig

    def plot_boxplot(self, column):
        fig = px.box(self.df, y=column, points="all", title=f"Boxplot of {column}")
        return fig

    def plot_scatter(self, x_col, y_col, color_col=None):
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, title=f"Scatter: {x_col} vs {y_col}")
        return fig

    def plot_line(self, x_col, y_col):
        fig = px.line(self.df, x=x_col, y=y_col, title=f"Line Chart: {x_col} vs {y_col}")
        return fig

    def plot_correlation_heatmap(self):
        numeric_df = self.df.select_dtypes(include='number')
        corr = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='Viridis'
        ))
        fig.update_layout(title='Correlation Heatmap')
        return fig

    def plot_parallel_coordinates(self, columns):
        fig = px.parallel_coordinates(self.df, dimensions=columns, title="Parallel Coordinates")
        return fig

    def plot_outliers_zscore(self, column, threshold=3):
        df_clean = self.df[[column]].dropna()
        df_clean['zscore'] = zscore(df_clean[column])
        df_clean['outlier'] = df_clean['zscore'].abs() > threshold
        fig = px.scatter(df_clean, x=df_clean.index, y=column, color='outlier', title=f"Outlier Detection ({column})")
        return fig

    def plot_time_series(self, time_col, value_col):
        fig = px.line(self.df.sort_values(time_col), x=time_col, y=value_col, title=f"Time Series of {value_col}")
        return fig

    def plot_pie_chart(self, category_col, values_col):
        fig = px.pie(self.df, names=category_col, values=values_col, title=f"Pie Chart of {category_col}")
        return fig

    def plot_bar_chart(self, category_col, values_col):
        fig = px.bar(self.df, x=category_col, y=values_col, title=f"Bar Chart of {category_col} vs {values_col}", text_auto=True)
        return fig

    def plot_stacked_bar(self, x_col, y_col, color_col):
        fig = px.bar(self.df, x=x_col, y=y_col, color=color_col, title=f"Stacked Bar: {x_col} vs {y_col} by {color_col}", text_auto=True)
        return fig

    def plot_consumption_by_license(self, license_col='نوع لایسنس', consumption_col='میزان مصرف', plot_type='box'):
        if plot_type == 'box':
            fig = px.box(self.df, x=license_col, y=consumption_col, points='all',
                         title='مقایسه مصرف آب بر اساس نوع لایسنس')
        elif plot_type == 'bar':
            grouped = self.df.groupby(license_col)[consumption_col].mean().reset_index()
            fig = px.bar(grouped, x=license_col, y=consumption_col,
                         title='میانگین مصرف آب بر اساس نوع لایسنس',
                         text_auto='.2s')
        else:
            raise ValueError("plot_type must be either 'box' or 'bar'")
        fig.update_layout(xaxis_title='نوع لایسنس', yaxis_title='میزان مصرف')
        return fig

    def describe_column(self, column):
        return self.df[column].describe()

    def show_missing_values(self):
        return self.df.isnull().sum().sort_values(ascending=False)

    def statistical_summary(self, column):
        data = self.df[column].dropna()
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25)
        }

    def scale_column(self, column, method='standard'):
        scaler = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }.get(method, StandardScaler())

        reshaped = self.df[[column]].dropna()
        scaled = scaler.fit_transform(reshaped)
        self.df[column + f'_{method}_scaled'] = pd.Series(scaled.flatten(), index=reshaped.index)
        return self.df[[column, column + f'_{method}_scaled']]

    def normalize_all_numeric(self, method='standard'):
        scaler = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }.get(method, StandardScaler())

        numeric_cols = self.df.select_dtypes(include='number').columns
        scaled_data = scaler.fit_transform(self.df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=[col + f'_{method}_scaled' for col in numeric_cols], index=self.df.index)
        self.df = pd.concat([self.df, scaled_df], axis=1)
        return self.df

