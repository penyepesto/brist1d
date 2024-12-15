import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.ensemble import StackingRegressor
import optuna
import re
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Step 1: Data Preprocessing
# Convert 'time' to datetime and extract time-based features
def extract_time_features(data):
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')
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

# Update target column name
target_col = 'bgplus1_00'

# Process activity data
activity_impact = {
    'Indoor climbing': -1,
    'Run': -1,
    'Strength training': -1,
    'Swim': -1,
    'Bike': -1,
    'Dancing': -1,
    'Stairclimber': -1,
    'Spinning': -1,
    'Walking': -1,
    'HIIT': -1,
    'Aerobic Workout': -1,
    'Hike': -1,
    'Zumba': -1,
    'Yoga': -1,
    'Swimming': -1,
    'Weights': -1,
    'Tennis': -1,
    'Sport': -1,
    'Workout': -1,
    'Outdoor Bike': -1,
    'Walk': -1,
    'Aerobic': -1,
    'Running': -1,
    'Strength': -1,
}

def process_activity_data(data):
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    for col in activity_columns:
        data[col] = data[col].map(activity_impact)
    data[activity_columns] = data[activity_columns].fillna(0)
    return data

df = process_activity_data(df)
dt = process_activity_data(dt)

# Function to get time offset in minutes
def get_time_offset_minutes(col_name):
    match = re.search(r'minus(\d+)_(\d+)', col_name)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        total_minutes = hours * 60 + minutes
        return total_minutes
    else:
        return None
print("step 1 is done")
# Step 2: Aggregate data into 15-minute intervals
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

# Step 3: Remove rows where more than half of the insulin values are zero
def remove_insulin_zero_rows(data):
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    # Count number of zero insulin values per row
    zero_counts = (data[insulin_cols] == 0).sum(axis=1)
    total_counts = data[insulin_cols].notnull().sum(axis=1)
    # Remove rows where more than half of the insulin values are zero
    data = data.loc[zero_counts <= (total_counts / 2)]
    return data.reset_index(drop=True)

df = remove_insulin_zero_rows(df)
print("step 3 is done")
# Advanced imputation using KNNImputer
from sklearn.impute import KNNImputer

def impute_missing_values(data):
    imputer = KNNImputer(n_neighbors=5)
    cols_to_impute = data.columns.difference(['id', 'p_num', 'time'])
    data[cols_to_impute] = imputer.fit_transform(data[cols_to_impute])
    return data

df = impute_missing_values(df)
dt = impute_missing_values(dt)

# Step 4: Create additional features based on relationships
def create_additional_features(data):
    # Difference in bg over time
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    bg_cols_sorted = sorted(bg_cols, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for i in range(len(bg_cols_sorted)-1):
        data[f'bg_diff_{i}'] = data[bg_cols_sorted[i]] - data[bg_cols_sorted[i+1]]

    # Moving averages and standard deviations
    window_sizes = [2, 4]  # Adjust window sizes as needed
    for window in window_sizes:
        data[f'bg_ma_{window}'] = data[bg_cols_sorted].rolling(window=window, axis=1).mean().iloc[:, -1]
        data[f'bg_std_{window}'] = data[bg_cols_sorted].rolling(window=window, axis=1).std().iloc[:, -1]

    # Interaction terms
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    carbs_cols = [col for col in data.columns if col.startswith('carbs_')]
    data['insulin_sum'] = data[insulin_cols].sum(axis=1)
    data['carbs_sum'] = data[carbs_cols].sum(axis=1)
    data['insulin_carbs_ratio'] = data['insulin_sum'] / (data['carbs_sum'] + 1e-5)

    # Previous glucose levels
    data['bg_previous'] = data[bg_cols_sorted].iloc[:, -1]

    # Rate of change of glucose
    data['bg_rate_of_change'] = data[bg_cols_sorted].diff(axis=1).mean(axis=1)

    return data

df = create_additional_features(df)
dt = create_additional_features(dt)

# Prepare data for modeling
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - exclude_cols
all_columns = list(df_columns | dt_columns)

# Ensure columns in training and test data match
for col in all_columns:
    if col not in df.columns:
        df[col] = 0
    if col not in dt.columns:
        dt[col] = 0

X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
print("step 4 is done")


# Adım 5: Sabit Hiperparametrelerle Model Eğitimi
# Modelleri tanımlayın
def get_models():
    lgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 10,
        'num_leaves': 31,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    xgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 10,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'random_state': 42,
        'n_jobs': -1
    }

    cb_params = {
        'iterations': 2000,
        'learning_rate': 0.01,
        'depth': 10,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'logging_level': 'Silent'
    }

    lgb_model = LGBMRegressor(**lgb_params)
    xgb_model = XGBRegressor(**xgb_params)
    cb_model = CatBoostRegressor(**cb_params)

    estimators = [
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('cb', cb_model)
    ]

    return estimators

# Modelleri alın
estimators = get_models()

# StackingRegressor tanımlayın
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

# Grup K-fold cross-validation ile modeli değerlendirin
groups = df['p_num']
gkf = GroupKFold(n_splits=5)
rmse_scores = []
for train_idx, val_idx in gkf.split(X_scaled, y, groups):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    stacking_regressor.fit(X_train, y_train)
    y_pred = stacking_regressor.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    print(f'Fold RMSE: {rmse}')

print(f'Ortalama RMSE: {np.mean(rmse_scores)}')

# Tüm veriyle modeli eğitin
stacking_regressor.fit(X_scaled, y)

# Test verisinde tahmin yapın
predictions = stacking_regressor.predict(X_test_scaled)

# Gönderim dosyasını oluşturun
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_stacked_model.csv', index=False)
