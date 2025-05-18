# analyzer.py
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import zscore, skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

# Ignore warnings for cleaner output in the app
warnings.filterwarnings('ignore')

class InteractiveDataAnalyzer:
    """
    A class to perform interactive data analysis and visualization
    using pandas and plotly. Simpler version with outlier plotting.
    """
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        # Work on a copy
        self._df = df.copy()
        # Attempt to convert potential object columns that are dates/numbers
        for col in self._df.columns:
             try:
                  self._df[col] = pd.to_numeric(self._df[col], errors='coerce')
             except:
                  try:
                       self._df[col] = pd.to_datetime(self._df[col], errors='coerce')
                  except:
                       # Convert remaining object columns to string for consistency in plotting/display
                       if pd.api.types.is_object_dtype(self._df[col]) or pd.api.types.is_categorical_dtype(self._df[col]):
                            self._df[col] = self._df[col].astype(str)
                       pass # Keep as is if conversion fails


    def get_data(self):
        """Returns the internal DataFrame."""
        return self._df

    def get_columns(self):
         """Returns a list of all column names."""
         return self._df.columns.tolist()

    def get_numeric_columns(self):
        """Returns a list of numeric column names."""
        numeric_cols = [col for col in self._df.columns if pd.api.types.is_numeric_dtype(self._df[col])]
        return numeric_cols

    def get_categorical_columns(self):
        """Returns a list of categorical (object/string/bool) column names."""
        categorical_cols = [col for col in self._df.columns if pd.api.types.is_object_dtype(self._df[col]) or pd.api.types.is_categorical_dtype(self._df[col]) or pd.api.types.is_bool_dtype(self._df[col])]
        return categorical_cols

    def get_datetime_columns(self):
        """Returns a list of datetime columns."""
        datetime_cols = [col for col in self._df.columns if pd.api.types.is_datetime64_any_dtype(self._df[col])]
        return datetime_cols


    def statistical_summary(self, column):
        """Returns basic descriptive statistics for a column."""
        if column not in self._df.columns:
            return "ستون یافت نشد."
        if column not in self.get_numeric_columns():
            return "ستون عددی نیست."
        return self._df[column].describe()

    def detailed_statistical_summary(self, column):
        """Returns detailed descriptive statistics including skewness and kurtosis."""
        if column not in self._df.columns:
            return "ستون یافت نشد."
        if column not in self.get_numeric_columns():
            return "ستون عددی نیست."

        col_data = self._df[column].dropna()
        if col_data.empty:
            return "ستون حاوی داده‌ای نیست."

        try:
            summary = {
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Standard Deviation': col_data.std(),
                'Variance': col_data.var(),
                'Skewness (کجی)': skew(col_data),
                'Kurtosis (کشیدگی)': kurtosis(col_data),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Count': col_data.count(),
                'IQR (دامنه میان چارکی)': col_data.quantile(0.75) - col_data.quantile(0.25)
            }
        except Exception as e:
            return f"خطا در محاسبه آماره تفصیلی: {e}"

        return pd.Series(summary, name=column)

    def show_missing_values(self):
        """Returns a DataFrame showing the count of missing values per column."""
        missing_counts = self._df.isnull().sum()
        missing_percentage = (self._df.isnull().sum() / len(self._df)) * 100
        missing_df = pd.DataFrame({
            'تعداد مقادیر گمشده': missing_counts,
            'درصد گمشده (%)': missing_percentage
        })
        # Filter to show only columns with missing values and sort
        missing_df = missing_df[missing_df['تعداد مقادیر گمشده'] > 0].sort_values(by='تعداد مقادیر گمشده', ascending=False)
        return missing_df

    def find_negative_values(self, column):
        """Finds and returns rows where the specified numeric column has negative values."""
        if column not in self._df.columns or column not in self.get_numeric_columns():
            return pd.DataFrame(), "ستون یافت نشد یا عددی نیست."

        negative_rows = self._df[self._df[column] < 0].copy()
        if negative_rows.empty:
            return negative_rows, f"هیچ مقدار منفی در ستون '{column}' یافت نشد."
        else:
            return negative_rows, f"{len(negative_rows)} ردیف با مقدار منفی در ستون '{column}' یافت شد."


    # --- Plotting Methods (Simplified) ---

    def plot_histogram(self, column):
        """Generates a histogram for a numeric column."""
        if column not in self._df.columns or column not in self.get_numeric_columns():
            return None, "ستون یافت نشد یا عددی نیست."
        try:
            fig = px.histogram(self._df, x=column, title=f'توزیع {column}')
            return fig, None
        except Exception as e:
             return None, f"خطا در رسم هیستوگرام: {e}"


    def plot_boxplot(self, column):
        """Generates a boxplot for a numeric column."""
        if column not in self._df.columns or column not in self.get_numeric_columns():
            return None, "ستون یافت نشد یا عددی نیست."
        try:
            fig = px.box(self._df, y=column, title=f'باکس‌پلات {column}')
            return fig, None
        except Exception as e:
             return None, f"خطا در رسم باکس‌پلات: {e}"

    def plot_violin(self, column):
        """Generates a violin plot for a numeric column."""
        if column not in self._df.columns or column not in self.get_numeric_columns():
            return None, "ستون یافت نشد یا عددی نیست."
        try:
            fig = px.violin(self._df, y=column, title=f'ویولن‌پلات {column}')
            return fig, None
        except Exception as e:
             return None, f"خطا در رسم ویولن‌پلات: {e}"


    def plot_scatterplot(self, x_col, y_col, color_col=None):
        """Generates a scatter plot between two columns, optionally colored by a third column."""
        if x_col not in self._df.columns or y_col not in self._df.columns:
             return None, "ستون‌های X یا Y یافت نشدند."
        # Removed check for color_col existence to allow user to select any column for color
        try:
            fig = px.scatter(self._df, x=x_col, y=y_col, color=color_col,
                             title=f'نمودار پراکندگی: {y_col} در مقابل {x_col}' + (f' بر اساس {color_col}' if color_col else ''))
            return fig, None
        except Exception as e:
            return None, f"خطا در رسم نمودار پراکندگی: {e}"


    def plot_pairplot(self, columns=None): # Removed color_col
        """
        Generates a pair plot (scatter matrix) for multiple numeric columns.
        """
        numeric_cols_all = self.get_numeric_columns()
        if columns is None:
            # Limit columns for performance
            if len(numeric_cols_all) > 10:
                numeric_cols_to_plot = numeric_cols_all[:10]
            else:
                 numeric_cols_to_plot = numeric_cols_all
        else:
            numeric_cols_to_plot = [col for col in columns if col in numeric_cols_all]
            if not numeric_cols_to_plot:
                 return None, "هیچ ستون عددی معتبری برای رسم Pair Plot انتخاب نشد."
            # Ensure selected columns are actually in the dataframe


        if len(numeric_cols_to_plot) < 2:
            return None, "برای رسم Pair Plot حداقل به دو ستون عددی نیاز است."

        try:
             fig = px.scatter_matrix(self._df,
                                      dimensions=numeric_cols_to_plot,
                                      title='ماتریس پراکندگی (Pair Plot)')
             fig.update_layout(diagonal_visible=False)
             return fig, None
        except Exception as e:
             return None, f"خطا در رسم Pair Plot: {e}"


    def plot_average_monthly_trend(self, time_col, metric_col): # Removed group_by_col
        """
        Plots the average of a metric over a time column.
        Intended for data in long format like long_usage*.csv with 'month' column.
        """
        if time_col not in self._df.columns or metric_col not in self._df.columns:
            return None, "ستون زمان یا معیار یافت نشد."
        # Ensure time_col is datetime or can be sorted chronologically
        if time_col not in self.get_datetime_columns() and not (pd.api.types.is_object_dtype(self._df[time_col]) or pd.api.types.is_categorical_dtype(self._df[time_col])):
             return None, f"ستون زمان ('{time_col}') باید از نوع تاریخ یا دسته‌ای قابل مرتب‌سازی باشد."

        if metric_col not in self.get_numeric_columns():
             return None, "ستون معیار باید عددی باشد."

        group_cols = [time_col]
        df_agg = self._df.groupby(group_cols)[metric_col].mean().reset_index()

        # Sort explicitly by time_col, coercing errors for robustness
        try:
             df_agg[time_col] = pd.to_datetime(df_agg[time_col], errors='coerce')
             df_agg = df_agg.sort_values(by=time_col)
        except Exception as e:
             # st.warning moved to app.py
             df_agg = df_agg.sort_values(by=time_col) # Fallback to string sort

        try:
            fig = px.line(df_agg, x=time_col, y=metric_col,
                          title=f'روند میانگین {metric_col} در طول زمان ({time_col})')
            return fig, None
        except Exception as e:
            return None, f"خطا در رسم روند ماهانه: {e}"


    def plot_data_availability_trend(self, time_col='month', availability_col='Percentage of Available Data Points'): # Removed group_by_col
         """
         Plots the trend of average data availability percentage over time.
         Assumes the existence of 'month' and 'Percentage of Available Data Points' columns.
         Suitable for long_usage*.csv data.
         """
         if time_col not in self._df.columns or availability_col not in self._df.columns:
              return None, f"ستون‌های '{time_col}' یا '{availability_col}' یافت نشدند. این نمودار برای داده‌های ماهانه پردازش شده مناسب است."
         if availability_col not in self.get_numeric_columns():
              return None, f"ستون '{availability_col}' عددی نیست."

         group_cols = [time_col]
         df_agg = self._df.groupby(group_cols)[availability_col].mean().reset_index()
         # Sort explicitly by time_col, coercing errors for robustness
         try:
             df_agg[time_col] = pd.to_datetime(df_agg[time_col], errors='coerce')
             df_agg = df_agg.sort_values(by=time_col)
         except Exception as e:
             # st.warning moved to app.py
             df_agg = df_agg.sort_values(by=time_col) # Fallback


         try:
            fig = px.line(df_agg, x=time_col, y=availability_col,
                          title='روند میانگین درصد اطلاعات موجود در طول زمان')
            return fig, None
         except Exception as e:
             return None, f"خطا در رسم روند درصد موجودیت داده: {e}"


    def plot_consumption_vs_operating_hours(self, consumption_col='Consumption (m³)', hours_col='Operating Hours (h)'): # Removed group_by_col
         """
         Plots Consumption vs Operating Hours, highlighting the potential negative correlation issue.
         Assumes the existence of 'Consumption (m³)' and 'Operating Hours (h)' columns.
         Suitable for long_usage*.csv data.
         """
         if consumption_col not in self._df.columns or hours_col not in self._df.columns:
              return None, f"ستون‌های '{consumption_col}' یا '{hours_col}' یافت نشدند. این نمودار برای داده‌های ماهانه پردازش شده مناسب است."
         if consumption_col not in self.get_numeric_columns() or hours_col not in self.get_numeric_columns():
              return None, "ستون‌های مصرف و ساعت کارکرد باید عددی باشند."

         # Optional: Remove rows where both are NaN for cleaner plot
         df_plot = self._df.dropna(subset=[consumption_col, hours_col]).copy()

         try:
             fig = px.scatter(df_plot, x=hours_col, y=consumption_col,
                              title=f'نمودار پراکندگی: {consumption_col} در مقابل {hours_col}')
             return fig, None
         except Exception as e:
             return None, f"خطا در رسم نمودار پراکندگی مصرف-ساعت کارکرد: {e}"


    # --- Outlier Identification Methods (Kept from last version) ---

    def identify_outliers_zscore(self, column, threshold=3):
        """
        Identifies potential outliers based on Z-score.
        Returns a dataframe of identified outliers and a boolean series indicating outliers.
        """
        if column not in self._df.columns or column not in self.get_numeric_columns():
            return pd.DataFrame(), pd.Series(dtype=bool), "ستون یافت نشد یا عددی نیست."

        df_clean = self._df.dropna(subset=[column]) # Remove NaNs for Z-score calculation
        if df_clean.empty:
             return pd.DataFrame(), pd.Series(dtype=bool), "ستون حاوی داده‌ای برای تشخیص موارد پرت نیست."

        col_data = df_clean[column]
        if col_data.std() == 0 or len(col_data) < 2:
             return pd.DataFrame(), pd.Series(dtype=bool), "واریانس ستون صفر است یا تعداد نقاط داده کمتر از 2 است، نمی‌توان Z-score را محاسبه کرد."

        z_scores = np.abs(zscore(col_data))
        outlier_indices = df_clean.index[z_scores > threshold]

        is_outlier_series = self._df.index.isin(outlier_indices)

        return self._df.loc[outlier_indices], is_outlier_series, None


    def identify_outliers_iqr(self, column, factor=1.5):
        """
        Identifies potential outliers based on IQR (Interquartile Range).
        Returns a dataframe of identified outliers and a boolean series indicating outliers.
        """
        if column not in self._df.columns or column not in self.get_numeric_columns():
            return pd.DataFrame(), pd.Series(dtype=bool), "ستون یافت نشد یا عددی نیست."

        df_clean = self._df.dropna(subset=[column])
        if df_clean.empty:
             return pd.DataFrame(), pd.Series(dtype=bool), "ستون حاوی داده‌ای برای تشخیص موارد پرت نیست."

        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            return pd.DataFrame(), pd.Series(dtype=bool), "دامنه میان چارکی (IQR) صفر است، نمی‌توان موارد پرت را با IQR تشخیص داد."

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers = df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)].copy()

        is_outlier_series = self._df.index.isin(outliers.index)

        return outliers, is_outlier_series, None


    def plot_outliers_scatter(self, column, is_outlier_series): # Removed group_by_col
         """
         Generates a scatter plot for a column, coloring points based on whether they are outliers.
         """
         if column not in self._df.columns or column not in self.get_numeric_columns():
             return None, "ستون اصلی برای رسم موارد پرت یافت نشد یا عددی نیست."
         if not isinstance(is_outlier_series, pd.Series) or not is_outlier_series.index.equals(self._df.index):
              return None, "سری بولین موارد پرت نامعتبر است. لطفاً ابتدا تشخیص موارد پرت را اجرا کنید."

         df_plot = self._df.copy()
         df_plot['وضعیت مورد پرت'] = is_outlier_series.map({True: 'مورد پرت شناسایی شده', False: 'داده عادی'}).fillna('نامشخص (NaN)') # Handle NaNs in original data

         try:
             fig = px.scatter(df_plot, x=df_plot.index, y=column, color='وضعیت مورد پرت',
                              title=f'نمودار موارد پرت در {column}',
                              hover_data=[column, 'وضعیت مورد پرت'])

             # Customizing colors for clarity (ensure colors are defined for all unique statuses)
             unique_statuses = df_plot['وضعیت مورد پرت'].unique()
             color_map = {}
             if 'مورد پرت شناسایی شده' in unique_statuses: color_map['مورد پرت شناسایی شده'] = 'red'
             if 'داده عادی' in unique_statuses: color_map['داده عادی'] = 'blue'
             if 'نامشخص (NaN)' in unique_statuses: color_map['نامشخص (NaN)'] = 'gray' # Color for NaN points

             fig.update_traces(marker=dict(size=6, opacity=0.8)) # Adjust marker size and transparency
             # Ensure colors are mapped correctly
             fig.update_layout(colorway=[color_map[s] for s in sorted(color_map.keys())]) # Use sorted keys for consistent order

             return fig, None
         except Exception as e:
             return None, f"خطا در رسم نمودار پراکندگی موارد پرت: {e}"


    def plot_barchart_agg(self, category_col, numeric_col): # Removed group_by_col
        """
        Generates a bar chart showing the mean of a numeric column
        grouped by a categorical column. If numeric_col is None, shows counts.
        """
        if category_col not in self._df.columns:
            return None, "ستون دسته‌ای یافت نشد."

        if numeric_col and numeric_col not in self._df.columns:
             return None, "ستون عددی یافت نشد."
        if numeric_col and numeric_col not in self.get_numeric_columns():
             return None, "ستون عددی، از نوع عددی نیست."

        try:
            if numeric_col:
                 fig = px.bar(self._df, x=category_col, y=numeric_col,
                              title=f'میانگین {numeric_col} بر اساس {category_col}')
            else:
                 # Plot value counts for a single categorical column
                 temp_series = self._df[category_col].astype(str).value_counts().reset_index()
                 temp_series.columns = [category_col, 'تعداد']
                 fig = px.bar(temp_series, x=category_col, y='تعداد',
                              title=f'تعداد ردیف‌ها بر اساس {category_col}')

            return fig, None
        except Exception as e:
             return None, f"خطا در رسم نمودار میله‌ای: {e}"


    def plot_correlation_heatmap(self):
        """Generates a correlation heatmap for numeric columns."""
        numeric_df = self._df[self.get_numeric_columns()].copy()

        if numeric_df.empty:
            return None, "هیچ ستون عددی برای رسم نقشه حرارتی همبستگی وجود ندارد."

        numeric_df = numeric_df.loc[:, numeric_df.var().fillna(0) != 0]
        if numeric_df.empty or numeric_df.shape[1] < 2:
             return None, "پس از حذف ستون‌های با واریانس صفر یا فقط یک مقدار، هیچ ستون عددی کافی برای رسم نقشه حرارتی باقی نماند."

        try:
            corr_matrix = numeric_df.corr()
            corr_matrix = corr_matrix.fillna(0)

            fig = ff.create_annotated_heatmap(z=corr_matrix.values,
                                              x=list(corr_matrix.columns),
                                              y=list(corr_matrix.index),
                                              annotation_text=corr_matrix.round(2).values,
                                              showscale=True, colorscale='Viridis')

            fig.update_layout(title='نقشه حرارتی همبستگی',
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,
                              yaxis_autorange='reversed',
                              height=max(400, len(corr_matrix.index) * 40),
                              width=max(600, len(corr_matrix.columns) * 50)
                              )
            return fig, None
        except Exception as e:
            return None, f"خطا در رسم نقشه حرارتی همبستگی: {e}"


    def scale_column(self, column, method='standard'):
        """Scales a single numeric column. Returns a new DataFrame slice with the original and scaled column."""
        if column not in self._df.columns or column not in self.get_numeric_columns():
            return pd.DataFrame(), "ستون یافت نشد یا عددی نیست."

        temp_df = self._df[[column]].copy()
        data_to_scale = temp_df[[column]].dropna()

        if data_to_scale.empty:
            return pd.DataFrame(), "ستون حاوی داده‌ای برای مقیاس‌بندی نیست."

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return pd.DataFrame(), "روش مقیاس‌بندی نامعتبر."

        try:
            scaled_data = scaler.fit_transform(data_to_scale)
            temp_df.loc[data_to_scale.index, f'{column}_scaled_{method}'] = scaled_data.flatten()

            return temp_df, None
        except Exception as e:
            return pd.DataFrame(), f"خطا در اعمال مقیاس‌بندی: {e}"
