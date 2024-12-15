import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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

# Handle missing values and process activity data
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
    'HIIT': 0,
    'Aerobic Workout': -1,
    'Hike': -1,
    'Zumba': -1,
    'Yoga': -1,
    'Swimming': -1,
    'Weights': 1,
    'Tennis': 1,
    'Sport': 1,
    'Workout': 0,
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
    def extract_minutes(col_name):
        match = re.search(r'minus(\d+)_(\d+)', col_name)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            total_minutes = hours * 60 + minutes
            return total_minutes
        else:
            return None
    activity_cols_last_hour = [col for col in activity_columns if extract_minutes(col) is not None and extract_minutes(col) <= 60]
    data['activity_impact_sum'] = data[activity_cols_last_hour].sum(axis=1)
    return data

df = process_activity_data(df)
dt = process_activity_data(dt)

# Fill missing values using interpolation
def interpolate_missing_values(data):
    data.sort_values(['p_num', 'time'], inplace=True)
    data = data.groupby('p_num').apply(lambda group: group.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill'))
    return data.reset_index(drop=True)

df = interpolate_missing_values(df)
dt = interpolate_missing_values(dt)

# Feature Engineering
# Limit lags to 12 (1 hour)
def create_lag_features(data, lags, feature_cols):
    for col in feature_cols:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

lags = range(1, 13)
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = create_lag_features(df, lags, feature_cols)
dt = create_lag_features(dt, lags, feature_cols)

def create_extended_lag_features(data, lags, feature_cols):
    for col in feature_cols:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

# Extend lags up to 24 periods (assuming 5-minute intervals, 2 hours)
extended_lags = range(1, 25)
additional_feature_cols = ['hrminus0_00', 'stepsminus0_00', 'calsminus0_00']
df = create_extended_lag_features(df, extended_lags, additional_feature_cols)
dt = create_extended_lag_features(dt, extended_lags, additional_feature_cols)

def create_rolling_features(data, windows, feature_cols):
    for col in feature_cols:
        for window in windows:
            data[f'{col}_rolling_mean_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            data[f'{col}_rolling_std_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
            data[f'{col}_rolling_min_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
            data[f'{col}_rolling_max_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
    return data

rolling_windows = [12, 24, 36, 48]  # Rolling windows up to 4 hours
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = create_rolling_features(df, rolling_windows, feature_cols)
dt = create_rolling_features(dt, rolling_windows, feature_cols)

def time_since_last_event(data, event_cols):
    for col in event_cols:
        event_occurred = data[col] > 0
        data[f'time_since_last_{col}'] = event_occurred.groupby(data['p_num']).cumsum()
        data[f'time_since_last_{col}'] = data.groupby('p_num')[f'time_since_last_{col}'].fillna(method='ffill').fillna(0)
        data[f'time_since_last_{col}'] = data.groupby('p_num').cumcount() - data[f'time_since_last_{col}']
    return data

event_cols = ['insulinminus0_00', 'carbsminus0_00', 'activity_impact_sum']
df = time_since_last_event(df, event_cols)
dt = time_since_last_event(dt, event_cols)

 # Exponential decay model for insulin effect
def insulin_effect(data):
    decay_rate = 0.05  # Adjust based on domain knowledge
    data['insulin_effect'] = data.groupby('p_num')['insulinminus0_00'].transform(
        lambda x: x[::-1].cumsum()[::-1] * decay_rate
    )
    return data

df = insulin_effect(df)
dt = insulin_effect(dt) 
# Cap outliers using quantiles
def cap_outliers(data, cols, lower_quantile=0.01, upper_quantile=0.99):
    for col in cols:
        lower_bound = data[col].quantile(lower_quantile)
        upper_bound = data[col].quantile(upper_quantile)
        data[col] = np.clip(data[col], lower_bound, upper_bound)
    return data

cols_to_cap = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = cap_outliers(df, cols_to_cap)
dt = cap_outliers(dt, cols_to_cap)

# Assume insulin sensitivity varies by hour
def insulin_sensitivity_factor(hour):
    if 0 <= hour < 6:
        return 1.2
    elif 6 <= hour < 12:
        return 1.0
    elif 12 <= hour < 18:
        return 0.8
    else:
        return 1.0

df['insulin_sensitivity'] = df['hour'].apply(insulin_sensitivity_factor)
dt['insulin_sensitivity'] = dt['hour'].apply(insulin_sensitivity_factor)

# Adjust insulin effect based on sensitivity
df['adjusted_insulin'] = df['insulinminus0_00'] * df['insulin_sensitivity']
dt['adjusted_insulin'] = dt['insulinminus0_00'] * dt['insulin_sensitivity']

# Additional feature interactions
df['bg_insulin_interaction'] = df['bgminus0_00'] * df['insulinminus0_00']
dt['bg_insulin_interaction'] = dt['bgminus0_00'] * dt['insulinminus0_00']

# Interaction between glucose and carbs
df['bg_carbs_interaction'] = df['bgminus0_00'] * df['carbsminus0_00']
dt['bg_carbs_interaction'] = dt['bgminus0_00'] * dt['carbsminus0_00']

# Polynomial features for bgminus0_00
df['bg_squared'] = df['bgminus0_00'] ** 2
dt['bg_squared'] = dt['bgminus0_00'] ** 2


# Prepare data for modeling
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - exclude_cols
all_columns = list(df_columns | dt_columns)

for col in all_columns:
    if col not in df.columns:
        df[col] = 0
    if col not in dt.columns:
        dt[col] = 0

X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Impute remaining missing values
X = X.fillna(0)
X_test = X_test.fillna(0)

# Update feature selection after potential changes
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X, y)
feature_importances = temp_lgbm.feature_importances_

N = 200  # Adjust N as needed
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

X = df[selected_features]
X_test = dt[selected_features]

# Cross-validation with GroupKFold and early stopping
groups = df['p_num']
gkf = GroupKFold(n_splits=5)

best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.003,
    'max_depth': 10,
    'n_estimators': 2000,
    'num_leaves': 31,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'subsample': 1.0,
    'random_state': 42,
    'verbose': -1,
    'early_stopping_rounds' : 50
}

lgbm_model = LGBMRegressor(**best_params)

rmse_scores = []
for fold, (train_index, val_index) in enumerate(gkf.split(X, y, groups)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        #early_stopping_rounds=50,
        #verbose=50  # Adjust as needed
    )

    y_pred = lgbm_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    print(f'Fold {fold+1} RMSE: {rmse}')

print(f'Average RMSE from cross-validation: {np.mean(rmse_scores)}')

# Train final model on full data
lgbm_model.fit(
    X, y,
    eval_set=[(X, y)],
    #early_stopping_rounds=50,
    #verbose=50
)

# Make predictions on the test set
predictions = lgbm_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_with_activity_8.csv', index=False)
print("finally!!!")

residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()
