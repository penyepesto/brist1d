import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import re
import gc
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Step 1: Basic Data Preprocessing and Time Features
def extract_time_features(data):
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S', errors='coerce')
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    # Cyclical transformation
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

# Function to get time offset in minutes from column name
def get_time_offset_minutes(col_name):
    match = re.search(r'minus(\d+)_(\d+)', col_name)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        total_minutes = hours * 60 + minutes
        return total_minutes
    else:
        return None

print("Step 1 completed: Basic preprocessing and activity data processed.")

# Step 3: Aggregate Data into 15-minute intervals
def aggregate_to_15_min_intervals(data):
    feature_types = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
    for feature in feature_types:
        feature_cols = [col for col in data.columns if col.startswith(feature + 'minus')]
        # For each column, get time offset
        col_time_offsets = {}
        for col in feature_cols:
            offset = get_time_offset_minutes(col)
            if offset is not None:
                col_time_offsets[col] = offset
        # Group columns into 15-minute bins
        bins = np.arange(0, 360 + 15, 15)  # From 0 to 360 minutes (6 hours), step of 15
        bin_labels = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        bin_cols = {label: [] for label in bin_labels}
        for col, offset in col_time_offsets.items():
            for (start, end) in bin_labels:
                if start <= offset < end:
                    bin_cols[(start, end)].append(col)
                    break
        # Compute the average for each bin
        for (start, end), cols in bin_cols.items():
            if len(cols) > 0:
                data[f'{feature}_{start}_{end}'] = data[cols].mean(axis=1)
            else:
                data[f'{feature}_{start}_{end}'] = np.nan
        # Drop original columns to save memory
        data.drop(columns=feature_cols, inplace=True)
    return data

df = aggregate_to_15_min_intervals(df)
dt = aggregate_to_15_min_intervals(dt)
print("Step 2 completed: Data aggregated into 15-minute intervals.")

# Step 4: Remove rows where more than half of the insulin values are zero
def remove_insulin_zero_rows(data):
    insulin_cols = [col for col in data.columns if 'insulin_' in col]
    if not insulin_cols:
        return data
    # Count number of zero insulin values per row
    zero_counts = (data[insulin_cols] == 0).sum(axis=1)
    total_counts = data[insulin_cols].notnull().sum(axis=1)
    # Remove rows where more than half of the insulin values are zero
    data = data.loc[zero_counts <= (total_counts / 2)]
    return data.reset_index(drop=True)

df = remove_insulin_zero_rows(df)
print("Step 3 completed: Rows with mostly zero insulin values removed.")

# Step 5: Create Additional Time Series and Derived Features
def create_additional_features_in_batches(data, lag_step=30, max_lag_minutes=360, batch_size=3):
    # Sum of insulin and carbs over the entire timeframe
    insulin_cols = [col for col in data.columns if col.startswith('insulin_') and '_' in col]
    carbs_cols = [col for col in data.columns if col.startswith('carbs_') and '_' in col]
    data['insulin_sum'] = data[insulin_cols].sum(axis=1)
    data['carbs_sum'] = data[carbs_cols].sum(axis=1)
    data['insulin_carbs_ratio'] = data['insulin_sum'] / (data['carbs_sum'] + 1e-5)
    data['insulin_carbs_interaction'] = data['insulin_sum'] * data['carbs_sum']

    # Split the range of lags into batches to process in chunks
    lag_batches = range(lag_step, max_lag_minutes + lag_step, lag_step)
    for batch_start in range(0, len(lag_batches), batch_size):
        current_batch = lag_batches[batch_start:batch_start + batch_size]
        
        # Processing each lag batch
        for lag in current_batch:
            lag_col_suffix = f'{lag}'
            if lag_col_suffix.isdigit():
                for feature in ['bg', 'insulin', 'carbs']:
                    feature_cols = [col for col in data.columns if col.startswith(f'{feature}_')]
                    lag_shift = int(lag / 5)  # convert minutes to intervals of 5
                    for col in feature_cols:
                        data[f'{col}_lag_{lag}'] = data[col].shift(lag_shift)
        
        # Clear memory after each batch
        gc.collect()
        print(f'Processed lag batch: {current_batch}')
    
 # Rolling windows for bg, insulin, and carbs (1-hour and 2-hour windows, e.g.)
    rolling_windows = [12, 24]  # 12 intervals for 1 hour (5-minute intervals?), 24 for 2 hours
    feature_types_map = {
        'bg': [col for col in data.columns if col.startswith('bg_') and not '_lag_' in col],
        'insulin': [col for col in data.columns if col.startswith('insulin_') and not '_lag_' in col],
        'carbs': [col for col in data.columns if col.startswith('carbs_') and not '_lag_' in col]
    }
    for window in rolling_windows:
        for feature, cols in feature_types_map.items():
            if cols:
                data[f'{feature}_rolling_mean_{window}'] = data[cols].rolling(window=window, axis=1).mean().iloc[:, -1]
                data[f'{feature}_rolling_std_{window}'] = data[cols].rolling(window=window, axis=1).std().iloc[:, -1]
                data[f'{feature}_rolling_min_{window}'] = data[cols].rolling(window=window, axis=1).min().iloc[:, -1]
                data[f'{feature}_rolling_max_{window}'] = data[cols].rolling(window=window, axis=1).max().iloc[:, -1]
    
    # Exponential Moving Average (EMA) for bg
    def exponential_moving_average(values, alpha=0.3):
        ema = [values[0]] if len(values) > 0 else [0]
        for val in values[1:]:
            ema.append(alpha * val + (1 - alpha) * ema[-1])
        return ema[-1]

    bg_cols = [col for col in data.columns if col.startswith('bg_') and '_' in col and not '_lag_' in col]
    if bg_cols:
        data['bg_ema'] = data.apply(lambda row: exponential_moving_average([row[col] for col in bg_cols if pd.notnull(row[col])]), axis=1)

    # Calculate acceleration (second differences) for bg to track changes
    if len(bg_cols) > 2:
        data['bg_acceleration'] = data[bg_cols].diff(axis=1).diff(axis=1).mean(axis=1)
    else:
        data['bg_acceleration'] = 0

    # Calculate advanced glucose change metrics
    # E.g., average change in bg over the last hour
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
        if hour_changes_mean.size > 0:
            data['bg_avg_change_last_hour'] = hour_changes_mean
        else:
            data['bg_avg_change_last_hour'] = 0
    else:
        data['bg_avg_change_last_hour'] = 0

    return data

df = create_additional_features_in_batches(df)
dt = create_additional_features_in_batches(dt)

print("Step 4 completed: Additional time series and derived features created.")

# Step 6: Outlier Handling for Key Features
# We'll apply IQR-based clipping for certain features
def handle_outliers(data, columns, clip_factor=3.0):
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - clip_factor * IQR
            upper_bound = Q3 + clip_factor * IQR
            data[col] = np.clip(data[col], lower_bound, upper_bound)
    return data

outlier_columns = ['insulin_sum', 'carbs_sum', 'bg_ema', 'hr_mean', 'steps_sum']  # Add any important numeric columns
df = handle_outliers(df, outlier_columns)
dt = handle_outliers(dt, outlier_columns)
print("Step 5 completed: Outliers handled for key features.")

# Step 7: Category-Based Missing Data Imputation
# For categorical data, forward-fill or backward-fill can be used if appropriate
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
df[categorical_cols] = df[categorical_cols].fillna(method='ffill').fillna(method='bfill')
dt[categorical_cols] = dt[categorical_cols].fillna(method='ffill').fillna(method='bfill')

print("Step 6 completed: Category-based missing data imputation.")

# Step 8: Clustering (KMeans) to create cluster-based features
def add_cluster_feature(data, features_to_cluster, n_clusters=5, cluster_name='cluster_label'):
    # Perform KMeans clustering on specified features
    valid_data = data[features_to_cluster].dropna()
    if valid_data.empty:
        data[cluster_name] = np.nan
        return data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_data)
    data.loc[valid_data.index, cluster_name] = cluster_labels
    data[cluster_name].fillna(data[cluster_name].mode()[0], inplace=True)
    return data

features_to_cluster = ['insulin_sum', 'carbs_sum', 'activity_impact_sum', 'bg_ema']
df = add_cluster_feature(df, features_to_cluster)
dt = add_cluster_feature(dt, features_to_cluster)
print("Step 7 completed: KMeans clustering added as a new feature.")

# Step 9: Additional Time-Based Categorical Encoding
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
one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
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

print("Step 8 completed: Time of day categories created and encoded.")

# Step 10: Data Normalization (Scaling) Using RobustScaler for Key Features
scaling_columns = ['insulin_sum', 'carbs_sum', 'bg_ema', 'hr_mean', 'steps_sum', 'insulin_carbs_ratio', 'insulin_carbs_interaction']
robust_scaler = RobustScaler()
for col in scaling_columns:
    if col in df.columns:
        df[col] = robust_scaler.fit_transform(df[[col]])
    if col in dt.columns:
        dt[col] = robust_scaler.transform(dt[[col]])

print("Step 9 completed: Data normalization using RobustScaler for key features.")

# Step 11: Polynomial Feature Generation for Key Interacting Features
poly_features = ['insulin_sum', 'carbs_sum', 'bg_previous']
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
print("Step 10 completed: Polynomial features generated for key features.")
print(df.shape)
print(dt.shape)
df.to_csv('processed_train.csv', index=False)
dt.to_csv('processed_test.csv', index=False)

# Final Step: Model Training and Submission

# Prepare data for modeling
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - exclude_cols
all_columns = list(df_columns | dt_columns)

# Ensure columns in training and test data match
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan
    if col not in dt.columns:
        dt[col] = np.nan

X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Impute remaining missing values with median
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

# Cross-Validation with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.01,
    'max_depth': 10,
    'n_estimators': 2000,
    'num_leaves': 31,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': -1
}

lgbm_model = LGBMRegressor(**best_params)

rmse_scores = []
for fold, (train_index, val_index) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50)]
    )

    y_pred = lgbm_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    print(f'Fold {fold+1} RMSE: {rmse}')

print(f'Average RMSE: {np.mean(rmse_scores)}')

# Train final model on full data
lgbm_model.fit(
    X, y,
    eval_set=[(X, y)],
    callbacks=[early_stopping(stopping_rounds=50)]
)

# Make predictions on the test set
predictions = lgbm_model.predict(X_test)

# Clip predictions to a reasonable range
predictions = np.clip(predictions, 2.5, 25)

# Create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_revised2.csv', index=False)
print("Completed: submission_revised.csv generated.")
