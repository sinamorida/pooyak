# app.py
import streamlit as st
import pandas as pd
import os
import datetime # Import datetime for date inputs
import numpy as np # Import numpy for NaN handling
from analyzer import InteractiveDataAnalyzer # Import the updated class

st.set_page_config(layout="wide", page_title="داشبورد تحلیل مصرف آب", initial_sidebar_state="expanded")

st.title("داشبورد تحلیل و پیش‌پردازش مصرف آب")
st.markdown("این داشبورد امکان بارگذاری، بررسی و تحلیل داده‌های مصرف آب را فراهم می‌کند.")

# Use session state to store the dataframe and analyzer object across reruns
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = None
if 'file_names' not in st.session_state:
    st.session_state['file_names'] = {} # To store uploaded files with simple names
if 'selected_file_name' not in st.session_state:
    st.session_state['selected_file_name'] = None
# Add session state for filtered data and outlier results
if 'filtered_data' not in st.session_state:
     st.session_state['filtered_data'] = None
if 'outlier_results' not in st.session_state:
     st.session_state['outlier_results'] = {} # Store {column: {method: {df: ..., series: ...}}}


# --- Sidebar for Upload and Selection ---
st.sidebar.header("بارگذاری و انتخاب داده")

uploaded_files = st.sidebar.file_uploader(
    "فایل‌های داده خام (CSV یا Excel) را بارگذاری کنید:",
    type=['csv', 'xlsx'],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Use a simple name for the file in the selectbox
        simple_name = uploaded_file.name
        # Check if file is already in session state by name
        # Check both name and approximate size/hash to see if it's a "new" file
        # Simple check: use file name and size
        file_id = f"{simple_name}_{uploaded_file.size}" # Basic unique ID

        if file_id not in st.session_state.get('loaded_file_ids', {}):
            try:
                # Determine file type and read accordingly
                if simple_name.endswith('.csv'):
                    # Assuming processed CSVs like long_usage*.csv or *_clean_info.csv are flat
                    df_loaded = pd.read_csv(uploaded_file)
                    st.sidebar.success(f"فایل CSV '{simple_name}' با موفقیت بارگذاری شد.")
                elif simple_name.endswith('.xlsx'):
                    # Attempting to read with header=[0,1] as per docs for raw data,
                    # but analysis is built for flat structure. User should ideally upload cleaned CSVs.
                    # If original Excels are uploaded, only basic structure might be viewable or may cause errors.
                    try:
                         # Try reading with multiindex header first as per docs
                         df_loaded = pd.read_excel(uploaded_file, header=[0, 1])
                         st.sidebar.warning(f"فایل Excel '{simple_name}' با هدر MultiIndex بارگذاری شد. برخی تحلیل‌ها ممکن است نیاز به داده مسطح شده داشته باشند.")
                    except Exception as e_excel_multi:
                         # If MultiIndex read fails or is not applicable, try standard read
                         try:
                              df_loaded = pd.read_excel(uploaded_file)
                              st.sidebar.info(f"فایل Excel '{simple_name}' به صورت جدول عادی بارگذاری شد.")
                         except Exception as e_excel_single:
                              st.sidebar.error(f"خطا در خواندن فایل Excel '{simple_name}': {e_excel_single}")
                              df_loaded = None # Set df_loaded to None if reading fails

                if df_loaded is not None:
                     # Store using the simple name, associate with the unique ID
                     st.session_state['file_names'][simple_name] = df_loaded
                     if 'loaded_file_ids' not in st.session_state: st.session_state['loaded_file_ids'] = {}
                     st.session_state['loaded_file_ids'][file_id] = simple_name # Store ID -> simple name mapping


            except Exception as e:
                st.sidebar.error(f"خطا در خواندن فایل {simple_name}: {e}")
                # Clean up partially stored data if error occurred
                if simple_name in st.session_state['file_names']:
                     del st.session_state['file_names'][simple_name]


# Display selectbox if files are loaded
if st.session_state['file_names']:
    file_options = list(st.session_state['file_names'].keys())
    # Find the index of the currently selected file to maintain selection on rerun
    current_index = 0
    if st.session_state['selected_file_name'] in file_options:
         current_index = file_options.index(st.session_state['selected_file_name']) + 1 # +1 because of the None option

    selected_file_name = st.sidebar.selectbox(
        "یکی از فایل‌های بارگذاری شده را انتخاب کنید:",
        options=[None] + file_options, # Add None option
        index=current_index, # Set initial index
        key='file_select_box' # Add a key for stability
    )

    # Update session state *only if* a file was selected (not None) or if it changed
    if selected_file_name and (st.session_state['selected_file_name'] != selected_file_name):
         st.session_state['selected_file_name'] = selected_file_name
         st.session_state['data'] = st.session_state['file_names'][selected_file_name].copy() # Store a copy
         # Instantiate or update the analyzer
         st.session_state['analyzer'] = InteractiveDataAnalyzer(st.session_state['data'])
         # Clear previous filtered data and outlier results when file changes
         st.session_state['filtered_data'] = None
         st.session_state['outlier_results'] = {} # Clear outlier results
         st.rerun() # Rerun to update the main content based on the new file

    elif selected_file_name is None and st.session_state['selected_file_name'] is not None:
         # Case where user selected None explicitly
         st.session_state['selected_file_name'] = None
         st.session_state['data'] = None
         st.session_state['analyzer'] = None
         st.session_state['filtered_data'] = None
         st.session_state['outlier_results'] = {} # Clear outlier results
         st.rerun() # Rerun to clear content

    # Use the dataframe from state for analysis/display in main content
    df = st.session_state['data']

else:
    # No files uploaded yet
    df = None
    st.session_state['data'] = None
    st.session_state['analyzer'] = None
    st.session_state['selected_file_name'] = None
    st.session_state['filtered_data'] = None
    st.session_state['outlier_results'] = {}


# --- Main Content Area (Conditional on data being loaded) ---
if df is not None and st.session_state['analyzer'] is not None:
    st.header(f"تحلیل برای فایل: **{st.session_state['selected_file_name']}**")

    # --- Overview ---
    st.subheader("نمای کلی داده")
    st.write(f"ابعاد مجموعه داده: {df.shape[0]} ردیف و {df.shape[1]} ستون")
    st.write("نمونه‌ای از 5 ردیف اول:")
    st.dataframe(df.head())

    # --- Data Access and Filtering ---
    st.subheader("دسترسی و فیلتر کردن داده")
    st.write("زیرمجموعه‌ای از داده‌ها را بر اساس مقدار یا محدوده در یک ستون خاص مشاهده کنید.")

    all_cols = st.session_state['analyzer'].get_columns()
    selected_filter_col = st.selectbox(
        "ستونی را برای فیلتر کردن انتخاب کنید:",
        options=[None] + all_cols,
        key='filter_select_col'
    )

    filtered_df_display = None # DataFrame to display after filtering
    filter_message = ""

    if selected_filter_col:
        col_data_for_type = df[selected_filter_col]
        is_numeric = pd.api.types.is_numeric_dtype(col_data_for_type)
        is_categorical = pd.api.types.is_object_dtype(col_data_for_type) or pd.api.types.is_categorical_dtype(col_data_for_type) or pd.api.types.is_bool_dtype(col_data_for_type)
        is_datetime = pd.api.types.is_datetime64_any_dtype(col_data_for_type)

        st.write(f"نوع داده ستون انتخاب شده: **{col_data_for_type.dtype}**")

        filter_params = {} # Dictionary to hold filter values

        if is_numeric:
            st.write("فیلتر بر اساس محدوده عددی:")
            col_min_val = df[selected_filter_col].min()
            col_max_val = df[selected_filter_col].max()

            # Handle potential NaN in min/max calculation for empty or all-NaN columns
            min_value = st.number_input(f"حداقل مقدار ({selected_filter_col}):", value=col_min_val if pd.notna(col_min_val) else 0.0, format="%.2f", key='filter_min')
            max_value = st.number_input(f"حداکثر مقدار ({selected_filter_col}):", value=col_max_val if pd.notna(col_max_val) else 0.0, format="%.2f", key='filter_max')

            filter_params = {'min': min_value, 'max': max_value}

        elif is_categorical:
            st.write("فیلتر بر اساس مقدار(های) دسته‌ای:")
            unique_values_incl_nan = df[selected_filter_col].unique().tolist()
            # Handle potential NaN in unique values for display
            display_options = [str(v) for v in unique_values_incl_nan if pd.notna(v)]
            nan_option_display = "[NaN] - مقادیر گمشده"
            nan_present = any(pd.isna(unique_values_incl_nan))
            if nan_present:
                 display_options.append(nan_option_display)
            display_options.sort() # Sort for readability

            selected_cat_value_str = st.selectbox(
                f"مقدار دقیق برای فیلتر کردن:",
                options=[None] + display_options,
                key='filter_cat_value'
            )
            if selected_cat_value_str is not None:
                if selected_cat_value_str == nan_option_display:
                    filter_params = {'value': np.nan} # Use numpy.nan to represent missing value
                else:
                    # Find the original value from the unique list (handle type conversions if needed)
                    # This can be tricky if original values are mixed types (e.g., numbers stored as objects)
                    # A safer way is to filter the original DataFrame directly by the string representation,
                    # but that might not match original data types perfectly.
                    # Let's rely on the analyzer's __init__ to have attempted type conversion.
                    original_value = next((v for v in unique_values_incl_nan if str(v) == selected_cat_value_str), selected_cat_value_str)
                    filter_params = {'value': original_value}

        elif is_datetime:
             st.write("فیلتر بر اساس محدوده تاریخ:")
             col_min_date_raw = pd.to_datetime(df[selected_filter_col], errors='coerce').min()
             col_max_date_raw = pd.to_datetime(df[selected_filter_col], errors='coerce').max()

             # Provide default dates, handling cases where date column is empty or all NaT
             default_start = col_min_date_raw.date() if pd.notna(col_min_date_raw) else datetime.date.today()
             default_end = col_max_date_raw.date() if pd.notna(col_max_date_raw) else datetime.date.today()

             start_date = st.date_input("تاریخ شروع:", value=default_start, key='filter_start_date')
             end_date = st.date_input("تاریخ پایان:", value=default_end, key='filter_end_date')
             filter_params = {'start': start_date, 'end': end_date}

        else:
            st.warning("نوع داده این ستون در حال حاضر برای فیلتر کردن مستقیم پشتیبانی نمی‌شود.")


        if st.button("اعمال فیلتر و نمایش داده", key='apply_filter_btn'):
            if filter_params: # Ensure filter_params were populated based on column type
                 filtered_df_display, filter_message = st.session_state['analyzer'].filter_data(selected_filter_col, filter_params)
                 st.session_state['filtered_data'] = filtered_df_display # Store filtered data in state
            else:
                 filter_message = "پارامترهای فیلتر به درستی مشخص نشده‌اند."
                 st.session_state['filtered_data'] = None # Clear state if filtering failed

        # Display stored filtered data if available
        if st.session_state['filtered_data'] is not None:
             st.write(filter_message) # Display the count/message from the last filter operation
             st.dataframe(st.session_state['filtered_data'])
        elif filter_message: # Display error/warning messages if no data is in state but a message exists
             st.warning(filter_message)

    else:
         st.info("یک ستون را از لیست بالا برای فعال کردن گزینه‌های فیلتر انتخاب کنید.")


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
    numeric_cols = st.session_state['analyzer'].get_numeric_columns() # Refresh numeric columns list
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
            key='outlier_col_select' # Changed key
        )
        if outlier_col:
            outlier_method = st.selectbox(
                "روش تشخیص موارد پرت:",
                options=['Z-score', 'IQR'],
                key='outlier_method_select' # Changed key
            )

            outliers_df = pd.DataFrame()
            is_outlier_series = pd.Series(dtype=bool)
            outlier_message = ""
            find_outliers_button_key = f'find_outliers_btn_{outlier_method}' # Keep button key dynamic


            if outlier_method == 'Z-score':
                z_threshold = st.slider("آستانه Z-score:", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key='z_threshold_slider') # Changed key
                if st.button(f"یافتن موارد پرت (Z-score > {z_threshold})", key=find_outliers_button_key):
                    outliers_df, is_outlier_series, outlier_message = st.session_state['analyzer'].identify_outliers_zscore(outlier_col, threshold=z_threshold)
                    # Store results in session state
                    st.session_state['outlier_results'][outlier_col] = {
                        outlier_method: {'df': outliers_df, 'series': is_outlier_series}
                    }

            elif outlier_method == 'IQR':
                iqr_factor = st.slider("ضریب IQR:", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key='iqr_factor_slider') # Changed key
                if st.button(f"یافتن موارد پرت (ضریب IQR = {iqr_factor})", key=find_outliers_button_key):
                     outliers_df, is_outlier_series, outlier_message = st.session_state['analyzer'].identify_outliers_iqr(outlier_col, factor=iqr_factor)
                     # Store results in session state
                     st.session_state['outlier_results'][outlier_col] = {
                         outlier_method: {'df': outliers_df, 'series': is_outlier_series}
                     }

            # Display stored outlier results if available for the selected column and method
            if outlier_col in st.session_state['outlier_results'] and outlier_method in st.session_state['outlier_results'][outlier_col]:
                 stored_results = st.session_state['outlier_results'][outlier_col][outlier_method]
                 displayed_outliers_df = stored_results['df']
                 # Use the last message from the button click if available, otherwise a default
                 # This requires storing the message too, or regenerating it. Let's regenerate based on stored df size.
                 if not displayed_outliers_df.empty:
                      st.write(f"{len(displayed_outliers_df)} مورد پرت شناسایی شده با روش **{outlier_method}**: ")
                      st.dataframe(displayed_outliers_df)
                 elif outlier_message: # Display message if detection was run and returned one (e.g., no outliers, error)
                     st.info(outlier_message)
                 else: # Default message if nothing stored and no recent message
                      st.info(f"برای مشاهده موارد پرت با روش **{outlier_method}**، دکمه 'یافتن موارد پرت' را کلیک کنید.")

        else:
            st.info("یک ستون عددی برای تشخیص موارد پرت انتخاب کنید.")


    # --- Statistical Summary ---
    st.subheader("خلاصه آماری")
    numeric_cols = st.session_state['analyzer'].get_numeric_columns() # Refresh numeric columns list
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
             st.dataframe(detailed_summary.to_frame()) # Convert Series to DataFrame for display
    else:
        st.info("یک ستون عددی را از لیست بالا برای مشاهده خلاصه آماری انتخاب کنید.")


    # --- Visualization Section ---
    st.subheader("بصری‌سازی داده")

    # Get available columns for plotting selectors
    all_cols = st.session_state['analyzer'].get_columns()
    categorical_cols = st.session_state['analyzer'].get_categorical_columns() # Refresh categorical columns list
    numeric_cols = st.session_state['analyzer'].get_numeric_columns() # Refresh numeric columns list

    # --- Group By Option ---
    st.sidebar.header("گزینه‌های رسم نمودار") # Group plot options in sidebar
    group_by_col_plot = st.sidebar.selectbox(
        "ستون دسته‌ای برای گروه‌بندی (Group By - اختیاری):",
        options=[None] + categorical_cols,
        key='plot_group_by_col'
    )


    plot_type = st.selectbox(
        "نوع نمودار را انتخاب کنید:",
        options=[
            'هیچ‌کدام',
            'هیستوگرام (توزیع)',
            'باکس‌پلات (موارد پرت و چارک‌ها)',
            'ویولن‌پلات (توزیع و چگالی)',
            'نمودار پراکندگی (دو ستون)', # Needs two columns
            'ماتریس پراکندگی (Pair Plot)', # Needs multiple columns
            'نمودار پراکندگی موارد پرت', # Needs column used for outlier detection + results
            'روند میانگین ماهانه', # Needs time col ('month') and metric col
            'روند میانگین درصد اطلاعات موجود (ماهانه)', # Needs 'month', 'Percentage of Available Data Points'
            'مصرف در مقابل ساعت کارکرد', # Needs 'Consumption (m³)', 'Operating Hours (h)'
            'نمودار میله‌ای (تجمیع بر اساس دسته)' # Needs categorical and numeric
            # Correlation heatmap is separate as it doesn't use group_by
        ],
        key='plot_type_select'
    )

    # Conditional inputs based on plot type
    fig = None
    plot_error = None

    # Check if group_by_col is suitable for the plot type (e.g., not used for Scatter Matrix color unless specified)
    # For simplicity, we pass group_by_col to relevant plots and let them decide how to use it (usually color)


    if plot_type in ['هیستوگرام (توزیع)', 'باکس‌پلات (موارد پرت و چارک‌ها)', 'ویولن‌پلات (توزیع و چگالی)']:
        selected_plot_col = st.selectbox(f"ستونی (عددی) را برای {plot_type} انتخاب کنید:", options=[None] + numeric_cols, key='single_plot_col')
        if selected_plot_col:
            if plot_type == 'هیستوگرام (توزیع)':
                fig, plot_error = st.session_state['analyzer'].plot_histogram(selected_plot_col, group_by_col=group_by_col_plot)
            elif plot_type == 'باکس‌پلات (موارد پرت و چارک‌ها)':
                fig, plot_error = st.session_state['analyzer'].plot_boxplot(selected_plot_col, group_by_col=group_by_col_plot)
            elif plot_type == 'ویولن‌پلات (توزیع و چگالی)':
                fig, plot_error = st.session_state['analyzer'].plot_violin(selected_plot_col, group_by_col=group_by_col_plot)
        else:
            plot_error = f"برای رسم {plot_type}، یک ستون عددی را انتخاب کنید."

    elif plot_type == 'نمودار پراکندگی (دو ستون)':
        st.info("برای نمودار پراکندگی، دو ستون را انتخاب کنید. ستون دسته‌بندی برای رنگ‌بندی استفاده می‌شود.")
        all_cols_scatter = st.session_state['analyzer'].get_columns()
        col1 = st.selectbox("ستون محور X:", options=[None] + all_cols_scatter, key='scatter_x')
        col2 = st.selectbox("ستون محور Y:", options=[None] + all_cols_scatter, key='scatter_y')
        # Use the global group_by_col for color in scatter plot
        color_by = group_by_col_plot # Use the group by selection for color
        if col1 and col2:
             fig, plot_error = st.session_state['analyzer'].plot_scatterplot(col1, col2, color_col=color_by)
        else:
             plot_error = "برای رسم نمودار پراکندگی، ستون‌های X و Y را انتخاب کنید."

    elif plot_type == 'ماتریس پراکندگی (Pair Plot)':
         st.info("این نمودار روابط بین جفت ستون‌های عددی را نشان می‌دهد. ستون دسته‌بندی برای رنگ‌بندی استفاده می‌شود.")
         # Allow selecting specific columns or use all numeric (up to a limit)
         plot_all_numeric = st.checkbox("رسم برای همه ستون‌های عددی (ممکن است زمان‌بر باشد و تنها ۱۰ ستون اول)", value=True, key='pairplot_all_cols')
         cols_to_plot = None
         # Use the global group_by_col for color in pair plot
         color_by_pairplot = group_by_col_plot

         if not plot_all_numeric:
             # User selects columns
             numeric_cols_for_pairplot = st.session_state['analyzer'].get_numeric_columns()
             cols_to_plot = st.multiselect(
                 "ستون‌های عددی را برای Pair Plot انتخاب کنید:",
                 options=numeric_cols_for_pairplot,
                 default=numeric_cols_for_pairplot[:5] if numeric_cols_for_pairplot else [], # Default selection
                 key='pairplot_selected_cols'
             )
             if not cols_to_plot:
                  plot_error = "حداقل دو ستون عددی برای رسم Pair Plot را انتخاب کنید."
                  cols_to_plot = None # Ensure it's None if list is empty

         # Call the plot function
         if plot_all_numeric or (cols_to_plot and len(cols_to_plot) >= 2):
             fig, plot_error = st.session_state['analyzer'].plot_pairplot(columns=cols_to_plot, color_col=color_by_pairplot)
         elif not plot_all_numeric and (cols_to_plot is None or len(cols_to_plot) < 2):
              pass # Error message already set above if cols_to_plot was empty

    elif plot_type == 'نمودار پراکندگی موارد پرت':
         st.info("نقطه ها در این نمودار بر اساس وضعیت مورد پرت بودنشان (طبق تشخیص اخیر) رنگ‌بندی می‌شوند.")
         st.warning("توجه: برای استفاده از این نمودار، ابتدا باید تشخیص موارد پرت را در بخش 'بررسی کیفیت داده' اجرا کرده باشید.")
         # Need to know which column was used for outlier detection
         # And the results (the boolean series indicating outliers)
         # Retrieve the column used for outlier detection and the latest results from session state
         # The outlier detection section stores results in st.session_state['outlier_results'][column][method]

         available_outlier_columns = list(st.session_state['outlier_results'].keys())
         selected_outlier_plot_col = st.selectbox(
              "ستونی که برای آن تشخیص موارد پرت اجرا شده است را انتخاب کنید:",
              options=[None] + available_outlier_columns,
              key='outlier_plot_col'
         )

         if selected_outlier_plot_col:
              available_methods_for_col = list(st.session_state['outlier_results'][selected_outlier_plot_col].keys())
              selected_outlier_plot_method = st.selectbox(
                   f"نتایج کدام روش تشخیص موارد پرت ({selected_outlier_plot_col}):",
                   options=[None] + available_methods_for_col,
                   key='outlier_plot_method'
              )

              if selected_outlier_plot_method:
                   # Retrieve the boolean series from session state
                   is_outlier_series_to_plot = st.session_state['outlier_results'][selected_outlier_plot_col][selected_outlier_plot_method]['series']
                   # Call the plot function
                   # Pass the global group_by_col_plot here as well for potential faceting/color
                   fig, plot_error = st.session_state['analyzer'].plot_outliers_scatter(
                       selected_outlier_plot_col,
                       is_outlier_series_to_plot,
                       group_by_col=group_by_col_plot # Pass group by col
                   )
              else:
                   plot_error = "یک روش تشخیص موارد پرت را انتخاب کنید."
         else:
              plot_error = "برای رسم نمودار موارد پرت، ستونی را انتخاب کنید که برای آن تشخیص موارد پرت اجرا شده است."


    elif plot_type == 'روند میانگین ماهانه':
        st.info("این نمودار میانگین یک معیار را در طول زمان (ماه) نشان می‌دهد. مناسب برای داده‌های 'long_usage*.csv'. ستون دسته‌بندی برای رنگ‌بندی خطوط استفاده می‌شود.")
        # Assumes the time column is named 'month' as per docs for df_long
        time_col_option = 'month'
        metric_col_trend = st.selectbox("ستون معیار (عددی) را برای نمایش روند انتخاب کنید:", options=[None] + numeric_cols, key='trend_metric_col')
        if time_col_option in df.columns and metric_col_trend:
            # Pass the global group_by_col_plot here
            fig, plot_error = st.session_state['analyzer'].plot_average_monthly_trend(time_col_option, metric_col_trend, group_by_col=group_by_col_plot)
        else:
             missing = []
             if time_col_option not in df.columns: missing.append(f"ستون '{time_col_option}'")
             if not metric_col_trend: missing.append("یک ستون عددی (معیار)")
             plot_error = f"برای رسم روند میانگین ماهانه، نیاز به {' و '.join(missing)} دارید."
             if group_by_col_plot and group_by_col_plot not in df.columns:
                  plot_error += f" (ستون دسته‌بندی '{group_by_col_plot}' یافت نشد)"


    elif plot_type == 'روند میانگین درصد اطلاعات موجود (ماهانه)':
         st.info("این نمودار میانگین 'درصد اطلاعات موجود' را در طول زمان (ماه) نشان می‌دهد. مناسب برای داده‌های 'long_usage*.csv'. ستون دسته‌بندی برای رنگ‌بندی خطوط استفاده می‌شود.")
         time_col_option = 'month'
         availability_col_option = 'Percentage of Available Data Points' # As per docs
         # Pass the global group_by_col_plot here
         fig, plot_error = st.session_state['analyzer'].plot_data_availability_trend(time_col=time_col_option, availability_col=availability_col_option, group_by_col=group_by_col_plot)
         if plot_error and (time_col_option not in df.columns or availability_col_option not in df.columns):
             # Refine error message if columns are missing
             plot_error = f"برای این نمودار نیاز به ستون‌های '{time_col_option}' و '{availability_col_option}' دارید. لطفاً مطمئن شوید فایل 'long_usage*.csv' را بارگذاری کرده‌اید."
             if group_by_col_plot and group_by_col_plot not in df.columns:
                  plot_error += f" (ستون دسته‌بندی '{group_by_col_plot}' یافت نشد)"


    elif plot_type == 'مصرف در مقابل ساعت کارکرد':
         st.info("این نمودار رابطه بین 'مصرف (m³)' و 'ساعت کارکرد (h)' را نشان می‌دهد. ستون دسته‌بندی برای رنگ‌بندی نقاط استفاده می‌شود.")
         consumption_col_option = 'Consumption (m³)'
         hours_col_option = 'Operating Hours (h)'
         # Pass the global group_by_col_plot here
         fig, plot_error = st.session_state['analyzer'].plot_consumption_vs_operating_hours(consumption_col=consumption_col_option, hours_col=hours_col_option, group_by_col=group_by_col_plot)
         if plot_error and (consumption_col_option not in df.columns or hours_col_option not in df.columns):
             # Refine error message if columns are missing
             plot_error = f"برای این نمودار نیاز به ستون‌های '{consumption_col_option}' و '{hours_col_option}' دارید. لطفاً مطمئن شوید فایل 'long_usage*.csv' را بارگذاری کرده‌اید."
             if group_by_col_plot and group_by_col_plot not in df.columns:
                  plot_error += f" (ستون دسته‌بندی '{group_by_col_plot}' یافت نشد)"


    elif plot_type == 'نمودار میله‌ای (تجمیع بر اساس دسته)':
         st.info("نمایش میانگین یک ستون عددی بر اساس دسته‌بندی یک ستون دیگر، یا تعداد رکوردها بر اساس یک ستون دسته‌ای. ستون دسته‌بندی اختیاری برای گروه‌بندی/رنگ‌بندی استفاده می‌شود.")
         selected_categorical_col_bar = st.selectbox(
              "ستون دسته‌ای اصلی (محور X) را انتخاب کنید:",
              options=[None] + categorical_cols,
              key='bar_cat_col'
         )
         selected_numeric_col_bar = st.selectbox(
              "ستون عددی (محور Y - برای تجمیع) را انتخاب کنید (اختیاری):",
              options=[None] + numeric_cols,
              key='bar_num_col'
         )
         if selected_categorical_col_bar:
              # Pass the global group_by_col_plot here as the secondary grouping/color
              fig, plot_error = st.session_state['analyzer'].plot_barchart_agg(selected_categorical_col_bar, selected_numeric_col_bar, group_by_col=group_by_col_plot)
              if plot_error:
                   st.warning(plot_error)
                   fig = None # ensure fig is None if error occurred
                   plot_error = None # Clear the error message after displaying

              if fig is None and selected_categorical_col_bar and selected_numeric_col_bar is None and not plot_error:
                   # If no numeric column selected and no error from plot_barchart_agg with group_by
                   # Plot counts by the primary categorical column
                   st.info(f"ستون عددی انتخاب نشد. نمایش تعداد ردیف‌ها بر اساس '{selected_categorical_col_bar}'.")
                   try:
                       # Note: This simple count plot ignores group_by_col_plot.
                       # A more complex count plot grouped by two columns is handled within plot_barchart_agg now.
                       if group_by_col_plot is None:
                            counts = df[selected_categorical_col_bar].astype(str).value_counts().reset_index()
                            counts.columns = [selected_categorical_col_bar, 'تعداد']
                            fig = px.bar(counts, x=selected_categorical_col_bar, y='تعداد',
                                        title=f'تعداد ردیف‌ها بر اساس {selected_categorical_col_bar}')
                       # If group_by_col_plot was selected, the plot_barchart_agg should have handled the double count plot
                       # If we reached here, it means numeric_col was None AND group_by_col_plot was None or handled.
                       # The logic inside plot_barchart_agg covers the count cases now. If fig is still None,
                       # it means plot_barchart_agg returned None with an error, which was handled above.
                   except Exception as e:
                       plot_error = f"خطا در رسم نمودار تعداد برای ستون دسته‌ای: {e}"


         else:
              plot_error = "برای رسم نمودار میله‌ای، حداقل یک ستون دسته‌ای اصلی را انتخاب کنید."


    # --- Correlation Heatmap (does not use group_by) ---
    st.subheader("نقشه حرارتی همبستگی")
    if st.button("رسم نقشه حرارتی همبستگی", key='plot_heatmap_btn'):
        heatmap_fig, heatmap_error = st.session_state['analyzer'].plot_correlation_heatmap()
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        elif heatmap_error:
            st.warning(heatmap_error)
        else:
            st.info("هیچ ستون عددی مناسبی برای رسم نقشه حرارتی یافت نشد.")


    # Display the plot if generated (and not already displayed like heatmap or outlier table)
    if plot_type != 'نقشه حرارتی همبستگی' and fig:
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
        key='scaling_col_select' # Changed key
    )

    if scaling_column:
         scaling_method = st.selectbox(
             "روش مقیاس‌بندی را انتخاب کنید:",
             options=['standard', 'minmax', 'robust'],
             key='scaling_method_select' # Changed key
         )

         if st.button("اعمال مقیاس‌بندی و نمایش نتیجه", key='scale_button'):
              try:
                  scaled_df_part, scale_error = st.session_state['analyzer'].scale_column(scaling_column, scaling_method)
                  if scale_error:
                      st.warning(scale_error)
                  elif not scaled_df_part.empty:
                      st.write(f"نتایج مقیاس‌بندی برای ستون '{scaling_column}' با روش '{scaling_method}':")
                      st.dataframe(scaled_df_part.head()) # Display head of results
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