import numpy as np 
import pandas as pd


def read_data(path):
    usage = pd.read_excel(path,header = [0,1])
    info = pd.read_excel(path)
    usage = usage[usage.columns[15:]]
    info = info[info.columns[:15]]
    info = info.iloc[1:] 
    return usage,info
column_translations = {
    "Unnamed: 1_level_1" : 'Subscription Code',
    'کد اشتراک': 'Subscription Code',
    'نوع پروانه': 'License Type',
    'شهرستان': 'County',
    'سریال کنتور': 'Meter Serial',
    'سایز کنتور': 'Meter Size',
    'تاریخ نصب': 'Installation Date',
    'زمان آخرین اتصال': 'Last Connection Time',
    'مصرف بازه (m³)': 'Consumption in Period (m³)',
    'ساعت کارکرد بازه (h)': 'Operating Hours in Period (h)',
    'میانگین دبی بازه l/s': 'Average Flow Rate in Period (l/s)',
    'دبی l/s': 'Flow Rate (l/s)',
    'تعداد دبی منفی': 'Number of Negative Flows',
    'درصد دبی منفی': 'Percentage of Negative Flows',
    'ساعت کارکرد (h)': 'Operating Hours (h)',
    'مصرف (m³) (m³)': 'Consumption (m³)',
    'تعداد اطلاعات موجود': 'Number of Available Data Points',
}
def prep(df):
    months = df.columns.levels[0][:12]
    return months

path = "1401.xlsx"
usage,info = read_data(path)
months = prep(usage)

print(months)


