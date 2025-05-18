# app.py
import streamlit as st
import pandas as pd
import os
import datetime
import numpy as np
from analyzer import InteractiveDataAnalyzer

st.set_page_config(layout="wide", page_title="داشبورد تحلیل مصرف آب", initial_sidebar_state="expanded")

st.title("داشبورد تحلیل و پیش‌پردازش مصرف آب")
st.markdown("این داشبورد امکان بارگذاری، بررسی و تحلیل داده‌های مصرف آب را فراهم می‌کند.")

# Use session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = None
if 'file_names' not in st.session_state:
    st.session_state['file_names'] = {}
if 'selected_file_name' not in st.session_state:
    st.session_state['selected_file_name'] = None
# Keep outlier results storage
if 'outlier_results' not in st.session_state:
     st.session_state['outlier_results'] = {} # Store {column: {method: {df: ..., series: ...}}}


# --- Sidebar for Upload and Selection ---
st.sidebar.header("بارگذاری و انتخاب داده")

uploaded_files = st.sidebar.file_uploader(
    "فایل‌های داده (CSV یا Excel) را بارگذاری کنید:",
    type=['csv', 'xlsx'],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        simple_name = uploaded_file.name
        # Basic unique ID for file content
        try:
            file_content = uploaded_file.getvalue()
            file_hash = hash(file_content) # Simple hash
            file_id = f"{simple_name}_{file_hash}"
        except Exception as e:
            st.sidebar.warning(f"Could not hash file {simple_name}: {e}. Using name/size fallback.")
            file_id = f"{simple_name}_{uploaded_file.size}" # Fallback

        if file_id not in st.session_state.get('loaded_file_ids', {}):
            try:
                if simple_name.endswith('.csv'):
                    df_loaded = pd.read_csv(uploaded_file)
                    st.sidebar.success(f"فایل CSV '{simple_name}' با موفقیت بارگذاری شد.")
                elif simple_name.endswith('.xlsx'):
                    try:
                         df_loaded = pd.read_excel(uploaded_file, header=[0, 1])
                         st.sidebar.warning(f"فایل Excel '{simple_name}' با هدر MultiIndex بارگذاری شد. تحلیل‌ها برای داده‌های مسطح شده مناسب‌ترند.")
                    except:
                         try:
                              df_loaded = pd.read_excel(uploaded_file)
                              st.sidebar.info(f"فایل Excel '{simple_name}' به صورت جدول عادی بارگذاری شد.")
                         except Exception as e_excel:
                              st.sidebar.error(f"خطا در خواندن فایل Excel '{simple_name}': {e_excel}")
                              df_loaded = None

                if df_loaded is not None:
                     st.session_state['file_names'][simple_name] = df_loaded
                     if 'loaded_file_ids' not in st.session_state: st.session_state['loaded_file_ids'] = {}
                     st.session_state['loaded_file_ids'][file_id] = simple_name # Store ID -> simple name mapping
                     # Set this file as selected if it's the first one or user wants it
                     if st.session_state['selected_file_name'] is None or len(st.session_state['file_names']) == 1:
                          st.session_state['selected_file_name'] = simple_name
                          st.session_state['data'] = df_loaded.copy()
                          st.session_state['analyzer'] = InteractiveDataAnalyzer(st.session_state['data'])
                          st.session_state['outlier_results'] = {} # Clear outlier results on new file
                          st.rerun() # Rerun to show data immediately


            except Exception as e:
                st.sidebar.error(f"خطا در خواندن فایل {simple_name}: {e}")
                if simple_name in st.session_state['file_names']:
                     del st.session_state['file_names'][simple_name]


# Display selectbox if files are loaded
if st.session_state['file_names']:
    file_options = list(st.session_state['file_names'].keys())
    current_index = 0
    if st.session_state['selected_file_name'] in file_options:
         current_index = file_options.index(st.session_state['selected_file_name']) + 1

    selected_file_name_widget = st.sidebar.selectbox(
        "یکی از فایل‌های بارگذاری شده را انتخاب کنید:",
        options=[None] + file_options,
        index=current_index,
        key='file_select_box'
    )

    # Logic to handle file selection change
    if st.session_state['selected_file_name'] != selected_file_name_widget:
         st.session_state['selected_file_name'] = selected_file_name_widget
         if selected_file_name_widget:
              st.session_state['data'] = st.session_state['file_names'][selected_file_name_widget].copy()
              st.session_state['analyzer'] = InteractiveDataAnalyzer(st.session_state['data'])
              st.session_state['outlier_results'] = {} # Clear outlier results on file change
         else:
              st.session_state['data'] = None
              st.session_state['analyzer'] = None
              st.session_state['outlier_results'] = {} # Clear outlier results on selecting None
         st.rerun() # Trigger rerun to update main content

    # Use the dataframe from state
    df = st.session_state['data']

else:
    df = None
    st.session_state['data'] = None
    st.session_state['analyzer'] = None
    st.session_state['selected_file_name'] = None
    st.session_state['outlier_results'] = {}


# --- Main Content Area (Conditional on data being loaded) ---
if df is not None and st.session_state['analyzer'] is not None:
    st.header(f"تحلیل برای فایل: **{st.session_state['selected_file_name']}**")

    # --- Overview ---
    st.subheader("نمای کلی داده")
    st.write(f"ابعاد مجموعه داده: {df.shape[0]} ردیف و {df.shape[1]} ستون")
    st.write("نمونه‌ای از 5 ردیف اول:")
    st.dataframe(df.head())

    # --- Data Quality Checks ---
    st.subheader("بررسی کیفیت داده")

    # Missing Values
    st.write("#### مقادیر گمشده")
    missing_df = st.session_state['analyzer'].show_missing_values()
    if not missing_df.empty:
        st.dataframe(missing_df)
    else:
        st.info("هیچ مقدار گمشده‌ای در مجموعه داده یافت نشد.")

    # Negative Values
    st.write("#### مقادیر منفی در ستون‌های عددی")
    numeric_cols = st.session_state['analyzer'].get_numeric_columns()
    if numeric_cols:
         negative_check_col = st.selectbox(
             "ستون عددی را برای بررسی مقادیر منفی انتخاب کنید:",
             options=[None] + numeric_cols,
             key='negative_check_col'
         )
         if negative_check_col:
              if st.button(f"یافتن مقادیر منفی در '{negative_check_col}'", key='find_neg_btn'):
                   negative_rows_df, neg_message = st.session_state['analyzer'].find_negative_values(negative_check_col)
                   st.write(neg_message)
                   if not negative_rows_df.empty:
                        st.dataframe(negative_rows_df)
    else:
        st.info("هیچ ستون عددی برای بررسی مقادیر منفی وجود ندارد.")


    # Outlier Detection
    st.write("#### تشخیص موارد پرت (Outliers)")
    if numeric_cols:
        outlier_col = st.selectbox(
            "ستون عددی را برای تشخیص موارد پرت انتخاب کنید:",
            options=[None] + numeric_cols,
            key='outlier_col_select'
        )
        if outlier_col:
            outlier_method = st.selectbox(
                "روش تشخیص موارد پرت:",
                options=['Z-score', 'IQR'],
                key='outlier_method_select'
            )

            outliers_df_result = pd.DataFrame()
            is_outlier_series_result = pd.Series(dtype=bool)
            outlier_message = ""
            find_outliers_button_key = f'find_outliers_btn_{outlier_method}_{outlier_col}' # Make button key unique per column/method

            # Attempt to retrieve stored results first
            stored_results = st.session_state['outlier_results'].get(outlier_col, {}).get(outlier_method)
            if stored_results:
                 outliers_df_result = stored_results['df']
                 is_outlier_series_result = stored_results['series']
                 # Regenerate a message based on stored results if needed, or rely on display logic below

            # --- Button to run detection ---
            if st.button(f"یافتن موارد پرت با روش {outlier_method}", key=find_outliers_button_key):
                 if outlier_method == 'Z-score':
                      z_threshold = st.slider("آستانه Z-score:", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key='z_threshold_slider', disabled=True) # Disable slider on button click? Or move above button. Let's move slider logic above.
                      # Run detection
                      outliers_df_result, is_outlier_series_result, outlier_message = st.session_state['analyzer'].identify_outliers_zscore(outlier_col, threshold=z_threshold)
                 elif outlier_method == 'IQR':
                      iqr_factor = st.slider("ضریب IQR:", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key='iqr_factor_slider', disabled=True) # Disable slider on button click?
                      # Run detection
                      outliers_df_result, is_outlier_series_result, outlier_message = st.session_state['analyzer'].identify_outliers_iqr(outlier_col, factor=iqr_factor)

                 # Store results in session state after running
                 if outlier_col not in st.session_state['outlier_results']: st.session_state['outlier_results'][outlier_col] = {}
                 st.session_state['outlier_results'][outlier_col][outlier_method] = {
                     'df': outliers_df_result,
                     'series': is_outlier_series_result
                 }
                 # Rerun to display results (Streamlit reruns automatically on button click anyway)
                 # st.rerun() # Not strictly needed due to button click behavior

            # --- Sliders for thresholds (shown regardless of button click state) ---
            if outlier_method == 'Z-score':
                 z_threshold_display = st.slider("آستانه Z-score:", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key='z_threshold_slider_display')
            elif outlier_method == 'IQR':
                 iqr_factor_display = st.slider("ضریب IQR:", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key='iqr_factor_slider_display')
            # Note: These sliders display the value *currently set* in the widget.
            # The button click logic above *uses* the value from the widget state *when clicked*.

            # --- Display outlier results (df and plot) ---
            if not outliers_df_result.empty:
                 st.write(f"{len(outliers_df_result)} مورد پرت شناسایی شده با روش **{outlier_method}**: ")
                 st.dataframe(outliers_df_result)

                 # --- Plot Outliers Scatter ---
                 st.write("#### نمودار پراکندگی موارد پرت")
                 # Use the results (column, series) from the last detection run
                 plot_outlier_fig, plot_outlier_error = st.session_state['analyzer'].plot_outliers_scatter(
                     outlier_col,
                     is_outlier_series_result
                 )
                 if plot_outlier_fig:
                     st.plotly_chart(plot_outlier_fig, use_container_width=True)
                 elif plot_outlier_error:
                     st.warning(plot_outlier_error)

            elif outlier_message: # Display message if detection was run and returned one (e.g., no outliers, error)
                 st.info(outlier_message)
            else: # Default message if nothing stored and no recent message
                 st.info(f"برای مشاهده موارد پرت با روش **{outlier_method}**، دکمه 'یافتن موارد پرت با روش {outlier_method}' را کلیک کنید.")


    else:
        st.info("هیچ ستون عددی برای تشخیص موارد پرت وجود ندارد.")


    # --- Statistical Summary ---
    st.subheader("خلاصه آماری")
    numeric_cols = st.session_state['analyzer'].get_numeric_columns()
    selected_numeric_col_summary = st.selectbox(
        "ستون عددی را برای مشاهده خلاصه آماری انتخاب کنید:",
        options=[None] + numeric_cols,
        key='summary_col'
    )
    if selected_numeric_col_summary:
        st.write(f"خلاصه آماری پایه برای ستون: **{selected_numeric_col_summary}**")
        basic_summary = st.session_state['analyzer'].statistical_summary(selected_numeric_col_summary)
        if isinstance(basic_summary, str):
             st.warning(basic_summary)
        else:
             st.dataframe(basic_summary)

        st.write(f"خلاصه آماری تفصیلی برای ستون: **{selected_numeric_col_summary}**")
        detailed_summary = st.session_state['analyzer'].detailed_statistical_summary(selected_numeric_col_summary)
        if isinstance(detailed_summary, str):
             st.warning(detailed_summary)
        else:
             st.dataframe(detailed_summary.to_frame())
    else:
        st.info("یک ستون عددی را از لیست بالا برای مشاهده خلاصه آماری انتخاب کنید.")


    # --- Visualization Section ---
    st.subheader("بصری‌سازی داده")

    all_cols = st.session_state['analyzer'].get_columns()
    categorical_cols = st.session_state['analyzer'].get_categorical_columns()
    numeric_cols = st.session_state['analyzer'].get_numeric_columns()

    plot_type = st.selectbox(
        "نوع نمودار را انتخاب کنید:",
        options=[
            'هیچ‌کدام',
            'هیستوگرام (توزیع)',
            'باکس‌پلات (موارد پرت و چارک‌ها)',
            'ویولن‌پلات (توزیع و چگالی)',
            'نمودار پراکندگی (دو ستون)',
            'ماتریس پراکندگی (Pair Plot)',
            # Outlier scatter plot is now integrated into Outlier Detection section
            'روند میانگین ماهانه',
            'روند میانگین درصد اطلاعات موجود (ماهانه)',
            'مصرف در مقابل ساعت کارکرد',
            'نمودار میله‌ای (تجمیع بر اساس دسته)'
        ],
        key='plot_type_select'
    )

    fig = None
    plot_error = None

    if plot_type in ['هیستوگرام (توزیع)', 'باکس‌پلات (موارد پرت و چارک‌ها)', 'ویولن‌پلات (توزیع و چگالی)']:
        selected_plot_col = st.selectbox(f"ستونی (عددی) را برای {plot_type} انتخاب کنید:", options=[None] + numeric_cols, key='single_plot_col')
        if selected_plot_col:
            if plot_type == 'هیستوگرام (توزیع)':
                fig, plot_error = st.session_state['analyzer'].plot_histogram(selected_plot_col)
            elif plot_type == 'باکس‌پلات (موارد پرت و چارک‌ها)':
                fig, plot_error = st.session_state['analyzer'].plot_boxplot(selected_plot_col)
            elif plot_type == 'ویولن‌پلات (توزیع و چگالی)':
                fig, plot_error = st.session_state['analyzer'].plot_violin(selected_plot_col)
        else:
            plot_error = f"برای رسم {plot_type}، یک ستون عددی را انتخاب کنید."

    elif plot_type == 'نمودار پراکندگی (دو ستون)':
        st.info("برای نمودار پراکندگی، دو ستون را انتخاب کنید.")
        col1 = st.selectbox("ستون محور X:", options=[None] + all_cols, key='scatter_x')
        col2 = st.selectbox("ستون محور Y:", options=[None] + all_cols, key='scatter_y')
        color_by = st.selectbox("رنگ‌بندی بر اساس (اختیاری):", options=[None] + all_cols, key='scatter_color') # Keep color_by option
        if col1 and col2:
             fig, plot_error = st.session_state['analyzer'].plot_scatterplot(col1, col2, color_by)
        else:
             plot_error = "برای رسم نمودار پراکندگی، ستون‌های X و Y را انتخاب کنید."

    elif plot_type == 'ماتریس پراکندگی (Pair Plot)':
         st.info("این نمودار روابط بین جفت ستون‌های عددی را نشان می‌دهد (حداکثر 10 ستون اول).")
         # Simplified Pair Plot - no option to select columns or color by category here
         numeric_cols_for_pairplot = st.session_state['analyzer'].get_numeric_columns()
         if len(numeric_cols_for_pairplot) > 10:
             st.warning("تعداد ستون‌های عددی زیاد است. تنها 10 ستون اول برای Pair Plot استفاده می‌شوند.")
             cols_to_plot = numeric_cols_for_pairplot[:10]
         else:
             cols_to_plot = numeric_cols_for_pairplot

         if len(cols_to_plot) >= 2:
              fig, plot_error = st.session_state['analyzer'].plot_pairplot(columns=cols_to_plot)
         else:
              plot_error = "حداقل دو ستون عددی برای رسم Pair Plot نیاز است."


    elif plot_type == 'روند میانگین ماهانه':
        st.info("این نمودار میانگین یک معیار را در طول زمان (ماه) نشان می‌دهد. مناسب برای داده‌های 'long_usage*.csv'.")
        time_col_option = 'month'
        metric_col_trend = st.selectbox("ستون معیار (عددی) را برای نمایش روند انتخاب کنید:", options=[None] + numeric_cols, key='trend_metric_col')
        if time_col_option in df.columns and metric_col_trend:
            fig, plot_error = st.session_state['analyzer'].plot_average_monthly_trend(time_col_option, metric_col_trend)
        else:
             missing = []
             if time_col_option not in df.columns: missing.append(f"ستون '{time_col_option}'")
             if not metric_col_trend: missing.append("یک ستون عددی (معیار)")
             plot_error = f"برای رسم روند میانگین ماهانه، نیاز به {' و '.join(missing)} دارید."


    elif plot_type == 'روند میانگین درصد اطلاعات موجود (ماهانه)':
         st.info("این نمودار میانگین 'درصد اطلاعات موجود' را در طول زمان (ماه) نشان می‌دهد. مناسب برای داده‌های 'long_usage*.csv'.")
         time_col_option = 'month'
         availability_col_option = 'Percentage of Available Data Points'
         fig, plot_error = st.session_state['analyzer'].plot_data_availability_trend(time_col=time_col_option, availability_col=availability_col_option)
         if plot_error and (time_col_option not in df.columns or availability_col_option not in df.columns):
             plot_error = f"برای این نمودار نیاز به ستون‌های '{time_col_option}' و '{availability_col_option}' دارید. لطفاً مطمئن شوید فایل 'long_usage*.csv' را بارگذاری کرده‌اید."


    elif plot_type == 'مصرف در مقابل ساعت کارکرد':
         st.info("این نمودار رابطه بین 'مصرف (m³)' و 'ساعت کارکرد (h)' را نشان می‌دهد.")
         consumption_col_option = 'Consumption (m³)'
         hours_col_option = 'Operating Hours (h)'
         fig, plot_error = st.session_state['analyzer'].plot_consumption_vs_operating_hours(consumption_col=consumption_col_option, hours_col=hours_col_option)
         if plot_error and (consumption_col_option not in df.columns or hours_col_option not in df.columns):
             plot_error = f"برای این نمودار نیاز به ستون‌های '{consumption_col_option}' و '{hours_col_option}' دارید. لطفاً مطمئن شوید فایل 'long_usage*.csv' را بارگذاری کرده‌اید."


    elif plot_type == 'نمودار میله‌ای (تجمیع بر اساس دسته)':
         st.info("نمایش میانگین یک ستون عددی بر اساس دسته‌بندی یک ستون دیگر، یا تعداد رکوردها بر اساس یک ستون دسته‌ای.")
         selected_categorical_col_bar = st.selectbox(
              "ستون دسته‌ای (محور X) را انتخاب کنید:",
              options=[None] + categorical_cols,
              key='bar_cat_col'
         )
         selected_numeric_col_bar = st.selectbox(
              "ستون عددی (محور Y - برای تجمیع) را انتخاب کنید (اختیاری):",
              options=[None] + numeric_cols,
              key='bar_num_col'
         )
         if selected_categorical_col_bar:
              fig, plot_error = st.session_state['analyzer'].plot_barchart_agg(selected_categorical_col_bar, selected_numeric_col_bar)
              if plot_error:
                   st.warning(plot_error)
                   fig = None
                   plot_error = None
         else:
              plot_error = "برای رسم نمودار میله‌ای، حداقل یک ستون دسته‌ای را انتخاب کنید."

    # Correlation Heatmap Button (placed outside the main plot_type if/elif for simplicity as it doesn't use common selectors)
    if plot_type == 'هیچ‌کدام': # Only show the heatmap button when 'None' is selected for main plots
         st.subheader("نقشه حرارتی همبستگی")
         if st.button("رسم نقشه حرارتی همبستگی", key='plot_heatmap_btn'):
             heatmap_fig, heatmap_error = st.session_state['analyzer'].plot_correlation_heatmap()
             if heatmap_fig:
                 st.plotly_chart(heatmap_fig, use_container_width=True)
             elif heatmap_error:
                 st.warning(heatmap_error)
             else:
                 st.info("هیچ ستون عددی مناسبی برای رسم نقشه حرارتی یافت نشد.")


    # Display the general plot if generated
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    if plot_error:
        st.warning(plot_error)


    # --- Optional Preprocessing / Scaling ---
    st.subheader("پیش‌پردازش اختیاری: مقیاس‌بندی داده")
    st.write("مقدارهای یک ستون عددی را با استفاده از روش‌های مختلف مقیاس‌بندی کنید و نتیجه را مشاهده کنید.")
    st.info("این بخش فقط برای نمایش است و ستون مقیاس‌بندی شده را به DataFrame اصلی اضافه نمی‌کند.")

    scaling_column = st.selectbox(
        "ستونی را برای مقیاس‌بندی انتخاب کنید:",
        options=[None] + numeric_cols,
        key='scaling_col_select'
    )

    if scaling_column:
         scaling_method = st.selectbox(
             "روش مقیاس‌بندی را انتخاب کنید:",
             options=['standard', 'minmax', 'robust'],
             key='scaling_method_select'
         )

         if st.button("اعمال مقیاس‌بندی و نمایش نتیجه", key='scale_button'):
              try:
                  scaled_df_part, scale_error = st.session_state['analyzer'].scale_column(scaling_column, scaling_method)
                  if scale_error:
                      st.warning(scale_error)
                  elif not scaled_df_part.empty:
                      st.write(f"نتایج مقیاس‌بندی برای ستون '{scaling_column}' با روش '{scaling_method}':")
                      st.dataframe(scaled_df_part.head())
                      st.write("...")
                  else:
                       st.info("مقیاس‌بندی انجام شد، اما داده‌ای برای نمایش وجود ندارد (ممکن است همه مقادیر گمشده باشند).")

              except Exception as e:
                   st.error(f"خطا در اعمال مقیاس‌بندی: {e}")

    else:
         st.info("برای استفاده از قابلیت مقیاس‌بندی، یک ستون عددی را انتخاب کنید.")

# --- Initial State Message ---
else:
    st.info("لطفاً فایل‌های داده خود (CSV یا Excel) را از نوار کناری بارگذاری کنید و سپس یکی را برای تحلیل انتخاب کنید.")
