import pandas as pd
import numpy as np
import warnings

# Suppress warnings if needed
# warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
FILE_PATH = '1401.xlsx'
NUM_METADATA_COLS = 15
NUM_COLS_PER_BLOCK = 8
NUM_MONTHLY_BLOCKS = 12

base_monthly_cols_english = [
    'FlowRate_LPS', 'NegFlowCount', 'NegFlowPerc', 'OpHours',
    'Consumption_M3', 'DataCount', 'ExpectedDataCount', 'DataAvailabilityPerc'
]

metadata_cols_english = [
    'UserName', 'UserID', 'UsageType_Persian', 'Province_Persian', 'City_Persian',
    'MeterOrAccountID', 'CapacityOrPipeSize', 'StartDate_Persian', 'LastReadingDateTime_Persian',
    'StatusFlag', 'TotalValue1_Unknown', 'TotalValue2_Unknown'
]

# --- Step 1: Load Data ---
try:
    df = pd.read_excel(FILE_PATH, header=[0, 1])
    print(f"Successfully loaded '{FILE_PATH}'. Initial Shape: {df.shape}")
except Exception as e:
    print(f"Failed to load file '{FILE_PATH}': {e}")
    exit()

expected_total_cols = NUM_METADATA_COLS + (NUM_MONTHLY_BLOCKS * NUM_COLS_PER_BLOCK)
if df.shape[1] != expected_total_cols:
    print(f"Warning: Expected {expected_total_cols} columns but found {df.shape[1]}.")

column_mapping = {}

for i in range(NUM_METADATA_COLS):
    column_mapping[df.columns[i]] = metadata_cols_english[i]

current_col_index = NUM_METADATA_COLS
for month_num in range(1, NUM_MONTHLY_BLOCKS + 1):
    for metric_name in base_monthly_cols_english:
        if current_col_index < df.shape[1]:
            old_col_name = df.columns[current_col_index]
            new_col_name = f"{metric_name}_{month_num}"
            column_mapping[old_col_name] = new_col_name
            current_col_index += 1
        else:
            print(f"Error: Ran out of columns in the DataFrame while trying to rename month {month_num}, metric '{metric_name}'.")
            exit()

df.rename(columns=column_mapping, inplace=True)
print("Columns renamed.")

cols_to_drop = [col for col in df.columns if 'Drop' in col]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns: {cols_to_drop}")
    metadata_cols_english = [col for col in metadata_cols_english if 'Drop' not in col]

print("\n--- Reshaping Data (Wide to Long) ---")
id_vars = metadata_cols_english

missing_id_vars = [var for var in id_vars if var not in df.columns]
if missing_id_vars:
    print(f"Error: Missing ID variables: {missing_id_vars}")
    exit()

stubnames = base_monthly_cols_english

df = df.reset_index(drop=True)
df['unique_id'] = df.index

try:
    df_long = pd.wide_to_long(
        df,
        stubnames=stubnames,
        i=['unique_id'] + id_vars,
        j='Month',
        sep='_',
        suffix='\d+'
    ).reset_index()
    df_long.drop(columns=['unique_id'], inplace=True)
    print("Reshaping successful.")
except Exception as e:
    print(f"Error during wide_to_long reshape: {e}")
    exit()

print("\n--- Converting Data Types (Long Format) ---")

numeric_cols_long = base_monthly_cols_english + ['CapacityOrPipeSize', 'TotalValue1_Unknown', 'TotalValue2_Unknown']
if 'UserID' in df_long.columns:
    numeric_cols_long.append('UserID')

for col in numeric_cols_long:
    if col in df_long.columns:
        df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found for numeric conversion.")

string_cols_long = ['UserName', 'UsageType_Persian', 'Province_Persian', 'City_Persian', 'MeterOrAccountID']
for col in string_cols_long:
    if col in df_long.columns:
        df_long[col] = df_long[col].astype(str)
    else:
        print(f"Warning: Column '{col}' not found for string conversion.")

bool_col = 'StatusFlag'
if bool_col in df_long.columns:
    df_long[bool_col] = df_long[bool_col].map({'False': False, 'True': True, False: False, True: True}).astype('boolean')
else:
    print(f"Warning: Column '{bool_col}' not found for boolean conversion.")

date_cols = ['StartDate_Persian', 'LastReadingDateTime_Persian']
for col in date_cols:
    if col in df_long.columns:
        df_long[col] = pd.to_datetime(df_long[col], errors='coerce')
        if df_long[col].isnull().any():
            print(f"Warning: Column '{col}' contains invalid dates. NaT introduced.")
    else:
        print(f"Warning: Column '{col}' not found for date conversion.")

print("\nData types after conversion (Long Format):")
print(df_long.info())

print("\n--- Missing Value Assessment (Long Format) ---")
missing_values_long = df_long.isnull().sum()
print("Missing values per column:")
print(missing_values_long[missing_values_long > 0])

if 'UsageType_Persian' in df_long.columns:
    df_long['UsageType_Persian'].fillna('Unknown', inplace=True)
    print("\nFilled missing 'UsageType_Persian' with 'Unknown'.")

print("\n--- Anomaly Identification (Long Format) ---")

metrics_to_check_neg = ['Consumption_M3', 'OpHours', 'FlowRate_LPS']
for col in metrics_to_check_neg:
    if col in df_long.columns:
        negative_rows = df_long[df_long[col] < 0]
        if not negative_rows.empty:
            print(f"\nIdentified {len(negative_rows)} rows with negative '{col}'.")

if 'NegFlowPerc' in df_long.columns:
    high_neg_flow = df_long[df_long['NegFlowPerc'] > 1]
    if not high_neg_flow.empty:
        print(f"\nIdentified {len(high_neg_flow)} rows with high 'NegFlowPerc'.")