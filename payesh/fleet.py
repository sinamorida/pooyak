import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# بخش اول: توابع رسم نمودار پایه (عمدتاً با Plotly Graph Objects)
# این توابع از پاسخ اولیه به درخواست شما برای رسم نمودارهای پایه هستند.
# -----------------------------------------------------------------------------

# 1.1 نمودار پراکندگی (Scatter Plot) - نسخه پایه با go
def plot_scatter_basic(x_data, y_data, title="نمودار پراکندگی (پایه)", x_label="محور X", y_label="محور Y", mode='markers', text_data=None, marker_size=None, marker_color=None):
    """
    رسم نمودار پراکندگی پایه.
    """
    fig = go.Figure(data=[go.Scatter(
        x=x_data,
        y=y_data,
        mode=mode,
        text=text_data,
        marker=dict(
            size=marker_size,
            color=marker_color,
            colorscale='Viridis',
            showscale=True if marker_color is not None and not isinstance(marker_color, str) else False
        )
    )])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        title_x=0.5
    )
    return fig

# 1.2. هیستوگرام (Histogram) - نسخه پایه با go
def plot_histogram_basic(data, title="هیستوگرام (پایه)", x_label="مقادیر", y_label="فراوانی", nbins=None):
    """
    رسم هیستوگرام پایه.
    """
    fig = go.Figure(data=[go.Histogram(
        x=data,
        nbinsx=nbins
    )])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        title_x=0.5
    )
    return fig

# 1.3. نمودار میله‌ای (Bar Chart) - نسخه پایه با go
def plot_bar_chart_basic(categories, values, title="نمودار میله‌ای (پایه)", x_label="دسته‌ها", y_label="مقادیر", orientation='v'):
    """
    رسم نمودار میله‌ای پایه.
    """
    if orientation == 'v':
        x_data, y_data = categories, values
    else:
        x_data, y_data = values, categories
        
    fig = go.Figure(data=[go.Bar(
        x=x_data,
        y=y_data,
        orientation=orientation
    )])
    fig.update_layout(
        title=title,
        xaxis_title=x_label if orientation == 'v' else y_label,
        yaxis_title=y_label if orientation == 'v' else x_label,
        title_x=0.5
    )
    return fig

# 1.4. نمودار میله‌ای پشته‌ای (Stacked Bar Chart) - نسخه پایه با go
def plot_stacked_bar_chart_basic(categories, data_dict, title="نمودار میله‌ای پشته‌ای (پایه)", x_label="دسته‌ها", y_label="مقادیر"):
    """
    رسم نمودار میله‌ای پشته‌ای پایه.
    data_dict: {'سری ۱': [مقادیر], 'سری ۲': [مقادیر]}
    """
    fig = go.Figure()
    for series_name, series_values in data_dict.items():
        fig.add_trace(go.Bar(
            x=categories,
            y=series_values,
            name=series_name
        ))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title_text='سری‌ها',
        title_x=0.5
    )
    return fig

# 1.5. نمودار جعبه‌ای (Box Plot) - نسخه پایه با go
def plot_box_plot_basic(data_list, names_list=None, title="نمودار جعبه‌ای (پایه)", y_label="مقادیر"):
    """
    رسم نمودار جعبه‌ای پایه.
    data_list: لیستی از آرایه‌ها، هر آرایه یک جعبه.
    """
    fig = go.Figure()
    if names_list is None:
        names_list = [f"جعبه {i+1}" for i in range(len(data_list))]
        
    for i, data_series in enumerate(data_list):
        fig.add_trace(go.Box(
            y=data_series,
            name=names_list[i]
        ))
        
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        title_x=0.5
    )
    return fig

# -----------------------------------------------------------------------------
# بخش دوم: توابع رسم نمودار پیشرفته (عمدتاً با Plotly Express)
# این توابع از پاسخی که برای تحلیل infos_df شما نوشته شد، آمده‌اند.
# -----------------------------------------------------------------------------

# توابع کمکی برای ایجاد داده نمونه (اگر لازم باشد، اما در اینجا از infos_df استفاده می‌کنیم)
def generate_sample_data_categorical(num_categories=5, num_values_per_cat=10):
    # ... (پیاده‌سازی از کد شما)
    categories = [f"دسته {chr(65+i)}" for i in range(num_categories)]
    data = []
    for cat in categories:
        for _ in range(num_values_per_cat):
            data.append({
                'دسته': cat,
                'مقدار_عددی': np.random.rand() * 100,
                'مقدار_دیگر': np.random.randint(1, 5),
                'زمان': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365), unit='D')
            })
    return pd.DataFrame(data)

def generate_time_series_data(num_series=3, num_points=50):
    # ... (پیاده‌سازی از کد شما)
    date_rng = pd.date_range(start='2023-01-01', periods=num_points, freq='D')
    df = pd.DataFrame(date_rng, columns=['تاریخ'])
    for i in range(num_series):
        df[f'سری {i+1}'] = np.random.randn(num_points).cumsum() + np.random.rand() * 20
    return df

# 2.1. نمودار پراکندگی (Scatter Plot) - با Plotly Express
def plot_scatter_px(df, x_col, y_col, color_col=None, size_col=None, symbol_col=None,
                    hover_name_col=None, hover_data_cols=None, facet_col=None, facet_row=None,
                    facet_col_wrap=None, log_x=False, log_y=False, trendline=None,
                    title="نمودار پراکندگی پیشرفته", x_label=None, y_label=None,
                    template="plotly_white", custom_hovertemplate=None):
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, size=size_col, symbol=symbol_col,
        hover_name=hover_name_col, hover_data=hover_data_cols, facet_col=facet_col,
        facet_row=facet_row, facet_col_wrap=facet_col_wrap, log_x=log_x, log_y=log_y,
        trendline=trendline, title=title,
        labels={x_col: x_label if x_label else x_col, y_col: y_label if y_label else y_col,
                color_col: color_col, size_col:size_col, symbol_col:symbol_col}, # افزودن لیبل برای این ستون‌ها
        template=template
    )
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    fig.update_layout(title_x=0.5); return fig

# 2.2. هیستوگرام (Histogram) - با Plotly Express
def plot_histogram_px(df, x_col, color_col=None, facet_col=None, facet_col_wrap=None,
                      nbins=None, histnorm=None, cumulative=False, barmode='stack',
                      title="هیستوگرام پیشرفته", x_label=None, y_label="فراوانی",
                      template="plotly_white", custom_hovertemplate=None):
    fig = px.histogram(
        df, x=x_col, color=color_col, facet_col=facet_col, facet_col_wrap=facet_col_wrap,
        nbins=nbins, histnorm=histnorm, cumulative=cumulative, barmode=barmode, title=title,
        labels={x_col: x_label if x_label else x_col, color_col: color_col},
        template=template
    )
    fig.update_layout(yaxis_title=y_label, title_x=0.5)
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    return fig

# 2.3. نمودار میله‌ای (Bar Chart) - با Plotly Express
def plot_bar_chart_px(df, x_col, y_col, color_col=None, barmode='group',
                      orientation='v', text_col=None,
                      title="نمودار میله‌ای پیشرفته", x_label=None, y_label=None,
                      template="plotly_white", custom_hovertemplate=None,
                      category_orders=None): # برای مرتب سازی دسته‌ها
    fig = px.bar(
        df, x=x_col, y=y_col, color=color_col, barmode=barmode, orientation=orientation,
        text=text_col, title=title,
        labels={x_col: x_label if x_label else x_col, y_col: y_label if y_label else y_col, color_col:color_col},
        template=template,
        category_orders=category_orders
    )
    if text_col: fig.update_traces(textposition='auto')
    fig.update_layout(title_x=0.5)
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    return fig

# 2.4. نمودار جعبه‌ای (Box Plot) - با Plotly Express
def plot_box_plot_px(df, x_col=None, y_col=None, color_col=None, notched=False,
                     points='outliers', title="نمودار جعبه‌ای پیشرفته", x_label=None, y_label=None,
                     template="plotly_white", custom_hovertemplate=None, order_map=None):
    labels_dict = {}
    if x_col: labels_dict[x_col] = x_label if x_label else x_col
    if y_col:
        if isinstance(y_col, list):
            for col_y_item in y_col: labels_dict[col_y_item] = y_label if y_label else col_y_item
        else: labels_dict[y_col] = y_label if y_label else y_col
    if color_col: labels_dict[color_col] = color_col

    fig = px.box(
        df, x=x_col, y=y_col, color=color_col, notched=notched, points=points, title=title,
        labels=labels_dict, template=template,
        category_orders=order_map # برای مرتب سازی x_col
    )
    fig.update_layout(title_x=0.5)
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    return fig

# 2.5. نمودار خطی (Line Plot) - با Plotly Express
def plot_line_px(df, x_col, y_cols, color_col_for_legend=None, line_group_col=None,
                 markers=False, line_shape='linear', title="نمودار خطی پیشرفته",
                 x_label=None, y_label=None, template="plotly_white", custom_hovertemplate=None):
    df_to_plot = df.copy() # کار روی کپی
    x_col_actual = x_col
    y_cols_actual = y_cols
    color_col_actual = color_col_for_legend
    labels_dict = {x_col: x_label if x_label else x_col}

    if isinstance(y_cols, list) and len(y_cols) > 0 :
        id_vars = [x_col_actual]
        if line_group_col and line_group_col != x_col_actual: id_vars.append(line_group_col)
        
        # اگر color_col_for_legend برای گروه‌بندی خطوط استفاده می‌شود و در id_vars نیست
        # و همچنین یک ستون معتبر در df است.
        if color_col_for_legend and color_col_for_legend in df_to_plot.columns and color_col_for_legend not in id_vars:
            id_vars.append(color_col_for_legend)
        
        df_melted = df_to_plot.melt(id_vars=list(set(id_vars)), value_vars=y_cols, var_name='سری_داده', value_name='مقدار_داده')
        y_cols_actual = 'مقدار_داده'
        
        if not color_col_for_legend or color_col_for_legend not in df_melted.columns :
             color_col_actual = 'سری_داده' # از نام ستون‌های y ذوب شده برای رنگ استفاده کن
        # else: color_col_actual = color_col_for_legend # اگر color_col_for_legend یک ستون مجزا برای رنگ است

        df_to_plot = df_melted
        labels_dict[y_cols_actual] = y_label if y_label else 'مقدار'
        if color_col_actual == 'سری_داده': labels_dict['سری_داده'] = "سری داده‌ها"
        elif color_col_actual and color_col_actual in df_to_plot.columns : labels_dict[color_col_actual] = color_col_actual
        
    elif isinstance(y_cols, str): # یک ستون y
        labels_dict[y_cols] = y_label if y_label else y_cols
        # color_col_actual باقی می‌ماند همان color_col_for_legend

    fig = px.line(
        df_to_plot, x=x_col_actual, y=y_cols_actual, color=color_col_actual,
        line_group=line_group_col, markers=markers, line_shape=line_shape,
        title=title, labels=labels_dict, template=template
    )
    fig.update_layout(title_x=0.5)
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    return fig

# 2.6. نمودار دایره‌ای (Pie Chart) - با Plotly Express
def plot_pie_px(df, names_col, values_col, color_col=None, hole=0,
                title="نمودار دایره‌ای/دونات پیشرفته", template="plotly_white",
                custom_hovertemplate=None, textinfo='percent+label'):
    fig = px.pie(
        df, names=names_col, values=values_col, color=color_col, hole=hole, title=title, template=template
    )
    fig.update_traces(textinfo=textinfo, textposition='inside')
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    fig.update_layout(title_x=0.5, legend_title_text=names_col if not color_col else color_col)
    return fig

# 2.7. هیت‌مپ (Heatmap) - با Plotly Graph Objects
def plot_heatmap_go(z_data, x_labels=None, y_labels=None, colorscale="Viridis",
                    showscale=True, title="هیت‌مپ", x_label="محور X", y_label="محور Y",
                    custom_hovertemplate=None):
    trace = go.Heatmap(
        z=z_data, x=x_labels, y=y_labels, colorscale=colorscale, showscale=showscale,
        hovertemplate=custom_hovertemplate if custom_hovertemplate else \
        "X: %{x}<br>Y: %{y}<br>مقدار: %{z}<extra></extra>"
    )
    layout = go.Layout(title=title, xaxis_title=x_label, yaxis_title=y_label, title_x=0.5)
    fig = go.Figure(data=[trace], layout=layout)
    return fig

# 2.8. نمودار سان‌برست (Sunburst Chart) - با Plotly Express
def plot_sunburst_px(df, path_cols, values_col, color_col=None, maxdepth=-1,
                     title="نمودار سان‌برست سلسله مراتبی", template="plotly_white",
                     custom_hovertemplate=None, color_continuous_scale=None):
    fig = px.sunburst(
        df, path=path_cols, values=values_col, color=color_col, maxdepth=maxdepth, title=title, template=template,
        color_continuous_scale=color_continuous_scale
    )
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    fig.update_layout(title_x=0.5); return fig

# 2.9. نمودار تری‌مپ (Treemap) - با Plotly Express
def plot_treemap_px(df, path_cols, values_col, color_col=None,
                    title="نمودار تری‌مپ سلسله مراتبی", template="plotly_white",
                    custom_hovertemplate=None, color_continuous_scale=None):
    fig = px.treemap(
        df, path=path_cols, values=values_col, color=color_col, title=title, template=template,
        color_continuous_scale=color_continuous_scale
    )
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    fig.update_layout(title_x=0.5); return fig

# 2.10. نمودار پراکندگی سه‌بعدی (3D Scatter Plot) - با Plotly Express
def plot_scatter_3d_px(df, x_col, y_col, z_col, color_col=None, symbol_col=None, size_col=None,
                       title="نمودار پراکندگی سه‌بعدی", template="plotly_white",
                       custom_hovertemplate=None):
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col, color=color_col, symbol=symbol_col, size=size_col,
        title=title, template=template, labels={x_col: x_col, y_col: y_col, z_col: z_col,
                                               color_col:color_col, symbol_col:symbol_col, size_col:size_col}
    )
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    fig.update_layout(title_x=0.5, scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col))
    return fig

# 2.11. نمودار سطحی سه‌بعدی (3D Surface Plot) - با Plotly Graph Objects
def plot_surface_3d_go(z_data, x_data=None, y_data=None, colorscale="Viridis",
                       title="نمودار سطحی سه‌بعدی", scene_aspectmode='cube'):
    if x_data is None and z_data.ndim > 1: x_data = np.arange(z_data.shape[1])
    if y_data is None and z_data.ndim > 1: y_data = np.arange(z_data.shape[0])
    
    if z_data.ndim > 1 and x_data is not None and y_data is not None:
        X, Y = np.meshgrid(x_data, y_data)
    else: # اگر z_data یک بعدی است یا x, y داده نشده
        X, Y = x_data, y_data


    trace = go.Surface(z=z_data, x=X, y=Y, colorscale=colorscale)
    layout = go.Layout(
        title=title, title_x=0.5,
        scene=dict(xaxis_title='محور X', yaxis_title='محور Y', zaxis_title='محور Z', aspectmode=scene_aspectmode)
    )
    fig = go.Figure(data=[trace], layout=layout); return fig

# 2.12. نمودار پراکندگی روی نقشه (Scatter Mapbox) - با Plotly Express
def plot_scatter_mapbox_px(df, lat_col, lon_col, color_col=None, size_col=None,
                           hover_name_col=None, hover_data_cols=None,
                           zoom=3, center=None, mapbox_style="open-street-map",
                           title="نمودار پراکندگی روی نقشه", template="plotly_white",
                           custom_hovertemplate=None):
    fig = px.scatter_mapbox(
        df, lat=lat_col, lon=lon_col, color=color_col, size=size_col,
        hover_name=hover_name_col, hover_data=hover_data_cols,
        zoom=zoom, center=center, title=title, template=template, height=600
    )
    fig.update_layout(mapbox_style=mapbox_style, margin={"r":0,"t":50,"l":0,"b":0}, title_x=0.5)
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    return fig

# 2.13. نمودار شمعی (Candlestick Chart) - با Plotly Graph Objects
def plot_candlestick_go(df, open_col, high_col, low_col, close_col, date_col,
                        title="نمودار شمعی قیمت سهام", template="plotly_white"):
    trace = go.Candlestick(
        x=df[date_col], open=df[open_col], high=df[high_col], low=df[low_col], close=df[close_col],
        increasing_line_color='green', decreasing_line_color='red'
    )
    layout = go.Layout(
        title=title, xaxis_title="تاریخ", yaxis_title="قیمت",
        xaxis_rangeslider_visible=True, template=template, title_x=0.5
    )
    fig = go.Figure(data=[trace], layout=layout); return fig

# 2.14. نمودار ویولن (Violin Plot) - با Plotly Express
def plot_violin_px(df, x_col=None, y_col=None, color_col=None, box=False, points='outliers',
                   title="نمودار ویولن", x_label=None, y_label=None,
                   template="plotly_white", custom_hovertemplate=None, order_map=None):
    labels_dict = {}
    if x_col: labels_dict[x_col] = x_label if x_label else x_col
    if y_col:
        if isinstance(y_col, list):
             for col_y_item in y_col: labels_dict[col_y_item] = y_label if y_label else col_y_item
        else: labels_dict[y_col] = y_label if y_label else y_col
    if color_col: labels_dict[color_col] = color_col
            
    fig = px.violin(
        df, x=x_col, y=y_col, color=color_col, box=box, points=points, title=title,
        labels=labels_dict, template=template, category_orders=order_map
    )
    fig.update_layout(title_x=0.5)
    if custom_hovertemplate: fig.update_traces(hovertemplate=custom_hovertemplate)
    return fig

# 2.15. جدول (Table) - با Plotly Graph Objects
def plot_table_go(df_or_header_values, cells_values=None, title="جدول داده‌ها"):
    if isinstance(df_or_header_values, pd.DataFrame):
        header_vals = list(df_or_header_values.columns)
        cell_vals = [df_or_header_values[col] for col in df_or_header_values.columns]
    else:
        header_vals = df_or_header_values
        cell_vals = cells_values
    trace = go.Table(
        header=dict(values=header_vals, fill_color='paleturquoise', align='left', font=dict(size=11)),
        cells=dict(values=cell_vals, fill_color='lavender', align='left', font=dict(size=10))
    )
    layout = go.Layout(title=title, title_x=0.5)
    fig = go.Figure(data=[trace], layout=layout); return fig


# -----------------------------------------------------------------------------
# بخش سوم: آماده‌سازی داده‌ها (DataFrame نمونه infos_df)
# -----------------------------------------------------------------------------
# ایجاد DataFrame نمونه مشابه با ساختار شما
data_dict_sample = {
    'Unnamed: 0': range(3888),
    'Subscription Code': [f'Sub_{i:04d}' for i in range(3888)],
    'License Type': np.random.choice(['خانگی', 'تجاری', 'صنعتی', None, 'کشاورزی', 'عمومی'], 3888, p=[0.5, 0.15, 0.05, 0.1, 0.1, 0.1]),
    'County': np.random.choice(['شهرستان الف', 'شهرستان ب', 'شهرستان ج', 'شهرستان د', 'شهرستان ه'], 3888),
    'Meter Serial': np.random.randint(100000, 999999, 3888).astype(float),
    'Meter Size': np.random.choice([0.5, 0.75, 1.0, 1.5, 2.0, 3.0], 3888, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05]),
    'Installation Date': pd.to_datetime('2018-01-01') + pd.to_timedelta(np.random.randint(0, 5*365), unit='D'),
    'Last Connection Time': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365*1.5), unit='D'), # دوره وسیع‌تر
    'Consumption in Period (m³)': np.abs(np.random.normal(loc=50, scale=70, size=3888)), # مصرف معمولا مثبت است
    'Operating Hours in Period (h)': np.abs(np.random.normal(loc=600, scale=150, size=3888)),
    'Average Flow Rate in Period (l/s)': np.abs(np.random.normal(loc=0.1, scale=0.08, size=3888))
}
infos_df = pd.DataFrame(data_dict_sample)
infos_df.loc[infos_df['Consumption in Period (m³)'] < 1, 'Consumption in Period (m³)'] = np.random.uniform(1,5, infos_df[infos_df['Consumption in Period (m³)'] < 1].shape[0]) # حداقل مصرف
infos_df.loc[infos_df['Operating Hours in Period (h)'] < 10, 'Operating Hours in Period (h)'] = np.random.uniform(10,50, infos_df[infos_df['Operating Hours in Period (h)'] < 10].shape[0])


# پر کردن مقادیر NaN و تبدیل انواع
infos_df['License Type'].fillna('نامشخص', inplace=True)
infos_df['Operating Hours in Period (h)'].fillna(infos_df['Operating Hours in Period (h)'].median(), inplace=True)
infos_df['Average Flow Rate in Period (l/s)'].fillna(infos_df['Average Flow Rate in Period (l/s)'].median(), inplace=True)

infos_df['Installation Date'] = pd.to_datetime(infos_df['Installation Date'])
infos_df['Last Connection Time'] = pd.to_datetime(infos_df['Last Connection Time'])

# ایجاد ستون‌های جدید
infos_df['Installation Year'] = infos_df['Installation Date'].dt.year
infos_df['Installation MonthName'] = infos_df['Installation Date'].dt.strftime('%Y-%m (%B)') # ماه با نام
infos_df['Days Since Last Connection'] = (pd.Timestamp.now() - infos_df['Last Connection Time']).dt.days
infos_df['Consumption per Hour (m³/h)'] = infos_df['Consumption in Period (m³)'] / infos_df['Operating Hours in Period (h)']
infos_df['Consumption per Hour (m³/h)'].replace([np.inf, -np.inf], 0, inplace=True) # مدیریت تقسیم بر صفر
infos_df['Consumption per Hour (m³/h)'].fillna(0, inplace=True)
infos_df['Meter Size Cat'] = infos_df['Meter Size'].astype(str) # برای نمودارهای دسته‌ای

# -----------------------------------------------------------------------------
# بخش چهارم: تست توابع (`if __name__ == "__main__":`)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42) # برای نتایج تکرارپذیر

    # --- تست توابع پایه (بخش اول) با داده‌های نمونه numpy ---
    print("--- شروع تست توابع پایه ---")
    scatter_x_basic = np.random.rand(30) * 5
    scatter_y_basic = 1.8 * scatter_x_basic + np.random.randn(30) * 1.5
    fig_scatter_b = plot_scatter_basic(scatter_x_basic, scatter_y_basic, title="نمودار پراکندگی پایه نمونه")
    fig_scatter_b.show()

    hist_data_basic = np.random.randn(300)
    fig_hist_b = plot_histogram_basic(hist_data_basic, title="هیستوگرام پایه نمونه", nbins=15)
    fig_hist_b.show()

    bar_cat_basic = ['محصول ۱', 'محصول ۲', 'محصول ۳']
    bar_val_basic = [25, 40, 32]
    fig_bar_b = plot_bar_chart_basic(bar_cat_basic, bar_val_basic, title="نمودار میله‌ای پایه نمونه")
    fig_bar_b.show()

    stacked_data_basic = {'فروش سال ۹۹': [10, 15, 12], 'فروش سال ۱۴۰۰': [12, 18, 10]}
    fig_stacked_b = plot_stacked_bar_chart_basic(bar_cat_basic, stacked_data_basic, title="نمودار میله‌ای پشته‌ای پایه")
    fig_stacked_b.show()

    box_data_b1 = np.random.normal(0, 1, 50)
    box_data_b2 = np.random.normal(2, 1.5, 50)
    fig_box_b = plot_box_plot_basic([box_data_b1, box_data_b2], names_list=['گروه الف', 'گروه ب'], title="نمودار جعبه‌ای پایه")
    fig_box_b.show()
    print("--- پایان تست توابع پایه ---")

    # --- تست توابع پیشرفته (بخش دوم) با داده‌های infos_df ---
    print("\n--- شروع رسم نمودارهای پیشرفته با داده‌های infos_df ---")

    # 0. نمایش نمونه‌ای از داده‌ها با جدول
    print("نمایش جدول نمونه از داده‌ها...")
    fig0 = plot_table_go(infos_df.sample(5, random_state=1).reset_index(drop=True), title="نمونه‌ای از داده‌های اشتراک (5 ردیف تصادفی)")
    fig0.show()

    # 1. نمودار پراکندگی پیشرفته
    print("نمایش نمودار پراکندگی: مصرف در مقابل ساعات کارکرد...")
    fig1 = plot_scatter_px(
        infos_df.sample(1000, random_state=1), # نمونه برای سرعت
        x_col='Operating Hours in Period (h)', y_col='Consumption in Period (m³)',
        color_col='License Type', size_col='Meter Size',
        hover_name_col='Subscription Code', hover_data_cols=['County', 'Average Flow Rate in Period (l/s)'],
        title="مصرف × ساعات کارکرد (رنگ: نوع مجوز، اندازه: سایز کنتور)",
        trendline='ols'
    )
    fig1.show()

    # 2. هیستوگرام پیشرفته
    print("نمایش هیستوگرام: توزیع مصرف بر اساس نوع مجوز...")
    fig2 = plot_histogram_px(
        infos_df, x_col='Consumption in Period (m³)', color_col='License Type',
        nbins=50, barmode='overlay', histnorm='probability density',
        title="توزیع چگالی احتمال مصرف بر اساس نوع مجوز"
    )
    fig2.update_traces(opacity=0.7)
    fig2.show()

    # 3. نمودار میله‌ای پیشرفته
    print("نمایش نمودار میله‌ای: میانگین مصرف بر اساس شهرستان...")
    mean_consum_county = infos_df.groupby('County')['Consumption in Period (m³)']\
                                 .mean().reset_index().sort_values('Consumption in Period (m³)', ascending=False)
    fig3 = plot_bar_chart_px(
        mean_consum_county, x_col='County', y_col='Consumption in Period (m³)', color_col='County',
        title="میانگین مصرف (متر مکعب) بر اساس شهرستان", text_col='Consumption in Period (m³)'
    )
    fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig3.show()
    
    print("نمایش نمودار میله‌ای پشته‌ای: تعداد اشتراک‌ها بر اساس شهرستان و نوع مجوز...")
    count_subs_stacked = infos_df.groupby(['County', 'License Type']).size().reset_index(name='تعداد')
    fig3_stacked = plot_bar_chart_px(
        count_subs_stacked, x_col='County', y_col='تعداد', color_col='License Type',
        barmode='stack', title="تعداد اشتراک‌ها: شهرستان (تفکیک نوع مجوز)"
    )
    fig3_stacked.show()


    # 4. نمودار جعبه‌ای پیشرفته
    print("نمایش نمودار جعبه‌ای: توزیع «مصرف به ازای ساعت کارکرد» بر اساس سایز کنتور...")
    # مرتب‌سازی سایز کنتورها برای نمایش بهتر
    sorted_meter_sizes = sorted(infos_df['Meter Size Cat'].unique(), key=float)
    order_map_meter_size = {'Meter Size Cat': sorted_meter_sizes}
    fig4 = plot_box_plot_px(
        infos_df, x_col='Meter Size Cat', y_col='Consumption per Hour (m³/h)',
        color_col='Meter Size Cat', points='outliers', notched=True,
        title="توزیع «مصرف/ساعت» بر اساس سایز کنتور",
        order_map=order_map_meter_size
    )
    fig4.show()

    # 5. نمودار خطی پیشرفته
    print("نمایش نمودار خطی: روند ماهانه نصب کنتورها بر اساس نوع مجوز...")
    installs_trend_license = infos_df.groupby([pd.Grouper(key='Installation Date', freq='M'), 'License Type'])\
                                     .size().reset_index(name='تعداد نصب')
    installs_trend_license = installs_trend_license.sort_values('Installation Date')
    fig5 = plot_line_px(
        installs_trend_license, x_col='Installation Date', y_cols='تعداد نصب',
        color_col_for_legend='License Type', # این ستون برای رنگ و لجند سری‌ها است
        markers=True, title="روند ماهانه نصب کنتورها (تفکیک نوع مجوز)"
    )
    fig5.show()

    # 6. نمودار دایره‌ای/دونات پیشرفته
    print("نمایش نمودار دونات: سهم انواع مجوز از کل تعداد اشتراک‌ها...")
    license_counts = infos_df['License Type'].value_counts().reset_index()
    license_counts.columns = ['License Type', 'تعداد']
    fig6 = plot_pie_px(
        license_counts, names_col='License Type', values_col='تعداد', hole=0.4,
        title="سهم انواع مجوز از کل اشتراک‌ها (دونات)"
    )
    fig6.show()

    # 7. هیت‌مپ پیشرفته
    print("نمایش هیت‌مپ: میانگین «روزهای گذشته از آخرین اتصال» (سال نصب × شهرستان)...")
    heatmap_data_days_conn = infos_df.groupby(['Installation Year', 'County'])['Days Since Last Connection']\
                                     .mean().unstack()
    fig7 = plot_heatmap_go(
        z_data=heatmap_data_days_conn.values,
        x_labels=heatmap_data_days_conn.columns.tolist(),
        y_labels=heatmap_data_days_conn.index.tolist(),
        colorscale='Plasma', title="میانگین «روز از آخرین اتصال» (سال نصب × شهرستان)",
        x_label="شهرستان", y_label="سال نصب"
    )
    fig7.show()

    # 8. سان‌برست و 9. تری‌مپ
    print("نمایش نمودار سان‌برست: تعداد اشتراک‌ها (شهرستان > نوع مجوز > سایز کنتور)...")
    # برای path در سان‌برست، مقادیر NaN مشکل‌ساز می‌شوند، پس آنها را با یک رشته جایگزین می‌کنیم اگر لازم باشد
    # اما در infos_df از قبل License Type و Meter Size Cat را مدیریت کردیم.
    # values باید مثبت باشند
    infos_df_sunburst = infos_df.copy()
    infos_df_sunburst['count'] = 1 # برای شمارش اشتراک‌ها
    fig8 = plot_sunburst_px(
        infos_df_sunburst, path_cols=['County', 'License Type', 'Meter Size Cat'],
        values_col='count', color_col='Meter Size', # رنگ بر اساس مقدار عددی سایز کنتور
        title="سلسله مراتب اشتراک‌ها: شهرستان > نوع مجوز > سایز کنتور",
        color_continuous_scale=px.colors.sequential.Blues # یک مقیاس رنگ برای color_col عددی
    )
    fig8.show()

    print("نمایش نمودار تری‌مپ: مجموع مصرف (نوع مجوز > شهرستان)...")
    fig9 = plot_treemap_px(
        infos_df[infos_df['Consumption in Period (m³)'] > 0], # فقط مقادیر مثبت برای values
        path_cols=['License Type', 'County'],
        values_col='Consumption in Period (m³)',
        color_col='Average Flow Rate in Period (l/s)', # رنگ بر اساس نرخ جریان
        title="مجموع مصرف: نوع مجوز > شهرستان (رنگ: نرخ جریان متوسط)",
        color_continuous_scale='RdYlGn'
    )
    fig9.show()
    
    # 10. نمودار پراکندگی سه‌بعدی
    print("نمایش نمودار پراکندگی سه‌بعدی: مصرف، ساعات کار، نرخ جریان...")
    fig10 = plot_scatter_3d_px(
        infos_df.sample(n=500, random_state=42), # نمونه کوچک برای سرعت
        x_col='Operating Hours in Period (h)',
        y_col='Average Flow Rate in Period (l/s)',
        z_col='Consumption in Period (m³)',
        color_col='License Type',
        size_col='Meter Size',
        title="مصرف، ساعات کار، نرخ جریان (رنگ: نوع مجوز، اندازه: سایز کنتور)"
    )
    fig10.update_traces(marker=dict(sizeref=0.05, sizemode='diameter')) # تنظیم بهتر اندازه
    fig10.show()

    # 14. نمودار ویولن
    print("نمایش نمودار ویولن: توزیع مصرف بر اساس شهرستان...")
    fig14 = plot_violin_px(
        infos_df, x_col='County', y_col='Consumption in Period (m³)',
        color_col='County', box=True, points=False, # points=False برای خوانایی بهتر با داده زیاد
        title="توزیع مصرف (متر مکعب) بر اساس شهرستان (ویولن)"
    )
    fig14.show()

    # نمودارهای نقشه و شمعی به داده‌های خاص نیاز دارند و در اینجا از آنها صرف‌نظر شده است.
    # نمودار سطحی سه‌بعدی نیز نیاز به آماده‌سازی داده پیچیده‌تری دارد.

    print("\n--- پایان نمایش تمام نمودارها ---")