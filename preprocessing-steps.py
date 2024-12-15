import pandas as pd
import numpy as np
import os
import gc
import re
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor , early_stopping
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Step 0: Load the data
df = pd.read_csv('/content/drive/MyDrive/brist1d-dataset/train.csv')
dt = pd.read_csv('/content/drive/MyDrive/brist1d-dataset/test.csv')

# Step 1: Basic Data Preprocessing and Time Features
def extract_time_features(data):
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S', errors='coerce')
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    # Cyclical transformation for hours and minutes
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 60)
    return data

df = extract_time_features(df)
dt = extract_time_features(dt)

# Clean column names
def clean_column_names(data):
    data.columns = data.columns.str.replace(':', '_', regex=False)
    data.columns = data.columns.str.replace('+', 'plus', regex=False)
    data.columns = data.columns.str.replace('-', 'minus', regex=False)
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return data

df = clean_column_names(df)
dt = clean_column_names(dt)

# Define target column
target_col = 'bgplus1_00'

# Step 2: Process Activity Data
activity_impact = {
    'Indoor climbing': 2,
    'Run': 3,
    'Strength training': 2,
    'Swim': 3,
    'Bike': 2,
    'Dancing': 2,
    'Stairclimber': 3,
    'Spinning': 3,
    'Walking': 1,
    'HIIT': 4,
    'Aerobic Workout': 2,
    'Hike': 2,
    'Zumba': 2,
    'Yoga': 1,
    'Swimming': 3,
    'Weights': 2,
    'Tennis': 2,
    'Sport': 2,
    'Workout': 2,
    'Outdoor Bike': 3,
    'Walk': 1,
    'Aerobic': 2,
    'Running': 3,
    'Strength': 2,
}

def process_activity_data(data):
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    for col in activity_columns:
        data[col] = data[col].map(activity_impact)
    data[activity_columns] = data[activity_columns].fillna(0)
    data['activity_impact_sum'] = data[activity_columns].sum(axis=1)
    # Create a feature for intense activity
    data['intense_activity_impact'] = data[activity_columns].apply(lambda row: sum(val for val in row if val >= 3), axis=1)
    return data

df = process_activity_data(df)
dt = process_activity_data(dt)

# Step 3: Aggregate Data into 15-minute intervals
def get_time_offset_minutes(col_name):
    match = re.search(r'minus(\d+)_(\d+)', col_name)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        total_minutes = hours * 60 + minutes
        return total_minutes
    else:
        return None

def aggregate_to_15_min_intervals(data):
    feature_types = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
    for feature in feature_types:
        feature_cols = [col for col in data.columns if col.startswith(feature + 'minus')]
        col_time_offsets = {}
        for col in feature_cols:
            offset = get_time_offset_minutes(col)
            if offset is not None:
                col_time_offsets[col] = offset
        bins = np.arange(0, 360 + 15, 15)  # From 0 to 360 minutes (6 hours), step of 15
        bin_labels = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        bin_cols = {label: [] for label in bin_labels}
        for col, offset in col_time_offsets.items():
            for (start, end) in bin_labels:
                if start <= offset < end:
                    bin_cols[(start, end)].append(col)
                    break
        for (start, end), cols in bin_cols.items():
            if len(cols) > 0:
                data[f'{feature}_{start}_{end}'] = data[cols].mean(axis=1)
            else:
                data[f'{feature}_{start}_{end}'] = np.nan
        data.drop(columns=feature_cols, inplace=True)
    return data

df = aggregate_to_15_min_intervals(df)
dt = aggregate_to_15_min_intervals(dt)

# Step 4: Remove rows where more than half of the insulin values are zero
def remove_insulin_zero_rows(data):
    insulin_cols = [col for col in data.columns if 'insulin_' in col]
    if not insulin_cols:
        return data
    zero_counts = (data[insulin_cols] == 0).sum(axis=1)
    total_counts = data[insulin_cols].notnull().sum(axis=1)
    data = data.loc[zero_counts <= (total_counts / 2)]
    return data.reset_index(drop=True)

df = remove_insulin_zero_rows(df)

# Step 5: Creating Summation Features for Insulin and Carbs
def create_sum_features(data):
    insulin_cols = [col for col in data.columns if 'insulin_' in col and '_' in col]
    carbs_cols = [col for col in data.columns if 'carbs_' in col and '_' in col]
    data['insulin_sum'] = data[insulin_cols].sum(axis=1)
    data['carbs_sum'] = data[carbs_cols].sum(axis=1)
    data['insulin_carbs_ratio'] = data['insulin_sum'] / (data['carbs_sum'] + 1e-5)
    data['insulin_carbs_interaction'] = data['insulin_sum'] * data['carbs_sum']
    return data

df = create_sum_features(df)
dt = create_sum_features(dt)

# Step 6: Creating Lag Features in Batches for Both Datasets
def create_lag_features_in_batches_save(data, dataset_name, lag_step=30, max_lag_minutes=360, batch_size=3):
    """
    Creates lag features in batches for the given dataset and saves each batch to disk.
    """
    total_batches = int(np.ceil((max_lag_minutes + lag_step - 1) / (batch_size * lag_step)))
    lag_feature_folder = f"lag_features_{dataset_name}"
    os.makedirs(lag_feature_folder, exist_ok=True)

    for batch_id in range(total_batches):
        batch_start_minutes = batch_id * batch_size * lag_step
        current_batch_lags = range(batch_start_minutes + lag_step, min(batch_start_minutes + (batch_size * lag_step) + 1, max_lag_minutes + lag_step), lag_step)
        batch_data = pd.DataFrame(index=data.index)  # Create a temporary DataFrame for the current batch
        batch_data['id'] = data['id'] # Include the 'id' column in the batch data

        for lag in current_batch_lags:
            for feature in ['bg', 'insulin', 'carbs']:
                feature_cols = [col for col in data.columns if col.startswith(feature + '_') and '_' in col and not col.endswith('_sum')]
                lag_shift = int(lag / 5)  # convert minutes to intervals of 5
                for col in feature_cols:
                    lagged_col_name = f'{col}_lag_{lag}'
                    if lagged_col_name not in batch_data.columns:
                        batch_data[lagged_col_name] = data[col].shift(lag_shift)
        batch_file_path = f"{lag_feature_folder}/lag_features_batch_{batch_id}.csv"
        batch_data.to_csv(batch_file_path, index=False) # Save without index
        del batch_data
        gc.collect()
        print(f'Processed and saved lag batch: {batch_id} for {dataset_name}')
    
    return data # Return the original DataFrame

df = create_lag_features_in_batches_save(df, 'train')
dt = create_lag_features_in_batches_save(dt, 'test')

def load_and_merge_lag_features(data, dataset_name):
    """
    Loads lag feature batches from disk and merges them into the original DataFrame.
    """
    lag_feature_folder = f"lag_features_{dataset_name}"
    for file in sorted(os.listdir(lag_feature_folder), key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1):
        if file.endswith(".csv"):
            lag_data_batch = pd.read_csv(f"{lag_feature_folder}/{file}") # Load without setting index_col
            data = pd.merge(data, lag_data_batch, on='id', how='left') # Merge on 'id' column
            del lag_data_batch
            gc.collect()
            print(f'Loaded and merged lag features from: {file} into {dataset_name}')
    return data
df = load_and_merge_lag_features(df, 'train')
dt = load_and_merge_lag_features(dt, 'test')

# Step 7: Creating Rolling Window Features
def create_rolling_features(data):
    rolling_windows = [12, 24]  # for 1 hour and 2 hours respectively if intervals are 5 minutes
    feature_types_map = {
        'bg': [col for col in data.columns if col.startswith('bg_') and not any(x in col for x in ['_lag_', '_sum', 'rolling', 'ema', 'acceleration', 'avg_change_last_hour'])],
        'insulin': [col for col in data.columns if col.startswith('insulin_') and not any(x in col for x in ['_lag_', '_sum', 'rolling'])],
        'carbs': [col for col in data.columns if col.startswith('carbs_') and not any(x in col for x in ['_lag_', '_sum', 'rolling'])]
    }
    for window in rolling_windows:
        for feature, cols in feature_types_map.items():
            rolling_mean = data[cols].mean(axis=1) if cols else 0
            rolling_std = data[cols].std(axis=1) if cols else 0
            rolling_min = data[cols].min(axis=1) if cols else 0
            rolling_max = data[cols].max(axis=1) if cols else 0
            data[f'{feature}_rolling_mean_{window}'] = rolling_mean
            data[f'{feature}_rolling_std_{window}'] = rolling_std
            data[f'{feature}_rolling_min_{window}'] = rolling_min
            data[f'{feature}_rolling_max_{window}'] = rolling_max
    return data

df = create_rolling_features(df)
dt = create_rolling_features(dt)

# Step 8: Exponential Moving Average (EMA) and Additional Features for bg
def create_additional_bg_features(data, alpha=0.3):
    bg_cols = [col for col in data.columns if col.startswith('bg_') and '_' in col and not '_lag_' in col and not col.endswith('_sum')]
    def exponential_moving_average(values, alpha):
        if len(values) == 0:
            return 0
        ema_value = values[0]
        for val in values[1:]:
            ema_value = alpha * val + (1 - alpha) * ema_value
        return ema_value

    if bg_cols:
        data['bg_ema'] = data.apply(lambda row: exponential_moving_average([row[col] for col in bg_cols if pd.notnull(row[col])], alpha), axis=1)
        if len(bg_cols) > 2:
            # Calculate acceleration (second differences)
            diff_first = data[bg_cols].diff(axis=1)
            diff_second = diff_first.diff(axis=1)
            data['bg_acceleration'] = diff_second.mean(axis=1).fillna(0)
        else:
            data['bg_acceleration'] = 0
        # Calculate advanced glucose change metrics
        hour_changes = []
        for i in range(12):  # last hour at 5-minute intervals
            column1 = f'bgminus0_{5*i}'
            column2 = f'bgminus0_{5*(i+1)}'
            if column1 in data.columns and column2 in data.columns:
                hour_changes.append(data[column1] - data[column2])
            else:
                hour_changes.append(0)
        if len(hour_changes) > 0:
            hour_changes_mean = np.nanmean(hour_changes, axis=0)
            data['bg_avg_change_last_hour'] = hour_changes_mean if hour_changes_mean.size else 0
        else:
            data['bg_avg_change_last_hour'] = 0
    else:
        data['bg_ema'] = 0
        data['bg_acceleration'] = 0
        data['bg_avg_change_last_hour'] = 0
    return data

df = create_additional_bg_features(df)
dt = create_additional_bg_features(dt)

# Step 9: Outlier Handling for Key Features
def handle_outliers(data, columns, clip_factor=3.0):
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - clip_factor * IQR
            upper_bound = Q3 + clip_factor * IQR
            data[col] = np.clip(data[col], lower_bound, upper_bound)
            gc.collect()  # Clear memory after each column is processed
    return data

outlier_columns = ['insulin_sum', 'carbs_sum', 'bg_ema', 'bg_acceleration', 'bg_avg_change_last_hour']
df = handle_outliers(df, outlier_columns)
dt = handle_outliers(dt, outlier_columns)

# Step 10: Category-Based Missing Data Imputation
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
df[categorical_cols] = df[categorical_cols].fillna(method='ffill').fillna(method='bfill')
dt[categorical_cols] = dt[categorical_cols].fillna(method='ffill').fillna(method='bfill')

# Step 11: Clustering (KMeans) to Create Cluster-Based Features
def add_cluster_feature(data, features_to_cluster, n_clusters=5, cluster_name='cluster_label'):
    valid_data = data[features_to_cluster].dropna()
    if valid_data.empty:
        data[cluster_name] = np.nan
        return data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_data)
    data.loc[valid_data.index, cluster_name] = cluster_labels
    data[cluster_name].fillna(data[cluster_name].mode()[0], inplace=True)
    del valid_data  # Clear temporary variable
    gc.collect()  # Collect garbage
    return data

features_to_cluster = ['insulin_sum', 'carbs_sum', 'activity_impact_sum', 'bg_ema']
df = add_cluster_feature(df, features_to_cluster)
dt = add_cluster_feature(dt, features_to_cluster)

# Step 12: Additional Time-Based Categorical Encoding
def add_time_of_day_category(data):
    conditions = [
        (data['hour'] >= 5) & (data['hour'] < 12),
        (data['hour'] >= 12) & (data['hour'] < 17),
        (data['hour'] >= 17) & (data['hour'] < 21),
        (data['hour'] >= 21) | (data['hour'] < 5)
    ]
    values = ['morning', 'afternoon', 'evening', 'night']
    data['time_of_day'] = np.select(conditions, values, default='unknown')
    return data

df = add_time_of_day_category(df)
dt = add_time_of_day_category(dt)

# Apply One-Hot Encoding for the time_of_day feature
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
time_of_day_train = df[['time_of_day']] if 'time_of_day' in df.columns else None
time_of_day_test = dt[['time_of_day']] if 'time_of_day' in dt.columns else None

if time_of_day_train is not None:
    encoded_time_of_day_df = pd.DataFrame(one_hot_encoder.fit_transform(time_of_day_train), index=df.index)
    encoded_time_of_day_df.columns = [f'time_of_day_{category}' for category in one_hot_encoder.categories_[0][1:]]
    df = pd.concat([df, encoded_time_of_day_df], axis=1)

if time_of_day_test is not None:
    encoded_time_of_day_dt = pd.DataFrame(one_hot_encoder.transform(time_of_day_test), index=dt.index)
    encoded_time_of_day_dt.columns = [f'time_of_day_{category}' for category in one_hot_encoder.categories_[0][1:]]
    dt = pd.concat([dt, encoded_time_of_day_dt], axis=1)

# Step 13: Data Normalization (Scaling) Using RobustScaler for Key Features
scaling_columns = ['insulin_sum', 'carbs_sum', 'bg_ema', 'insulin_carbs_ratio', 'insulin_carbs_interaction', 'activity_impact_sum', 'bg_acceleration', 'bg_avg_change_last_hour']
robust_scaler = RobustScaler()
for col in scaling_columns:
    if col in df.columns and col in dt.columns:
        # Fit RobustScaler on training data column
        df[col] = robust_scaler.fit_transform(df[[col]])
        # Transform test data using the same scaler
        dt[col] = robust_scaler.transform(dt[[col]])

# Step 14: Polynomial Feature Generation for Key Interacting Features
poly_features = ['insulin_sum', 'carbs_sum', 'activity_impact_sum']
poly = PolynomialFeatures(degree=2, include_bias=False)
df_poly = df[poly_features].copy()
dt_poly = dt[poly_features].copy()

df_poly_transformed = poly.fit_transform(df_poly)
dt_poly_transformed = poly.transform(dt_poly)

poly_feature_names = poly.get_feature_names_out(poly_features)
df_poly_df = pd.DataFrame(df_poly_transformed, columns=poly_feature_names, index=df.index)
dt_poly_df = pd.DataFrame(dt_poly_transformed, columns=poly_feature_names, index=dt.index)

df = pd.concat([df, df_poly_df], axis=1)
dt = pd.concat([dt, dt_poly_df], axis=1)

# Remove duplicates in column names if any
df = df.loc[:, ~df.columns.duplicated()]
dt = dt.loc[:, ~dt.columns.duplicated()]

print(f"Train DataFrame: {df.shape}")
print(f"Test DataFrame: {dt.shape}")

# Ensure that both df and dt have the same columns
for col in df.columns:
    if col not in dt.columns:
        dt[col] = np.nan
for col in dt.columns:
    if col not in df.columns:
        df[col] = np.nan

# Step 15: Final Data Preparation and Model Training
exclude_cols = {'id', 'time', target_col, 'p_num', 'time_of_day'} 
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - exclude_cols
all_columns = list(df_columns | dt_columns)
X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Impute remaining missing values with median
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)


# Ensure the shapes after processing
print("Final shapes:")
print(f"Train DataFrame: {df.shape}")
print(f"Test DataFrame: {dt.shape}")

# Save processed dataframes for debugging
df.to_csv('/content/drive/MyDrive/brist1d-dataset/processed_train.csv', index=False)
dt.to_csv('/content/drive/MyDrive/brist1d-dataset/processed_test.csv', index=False)