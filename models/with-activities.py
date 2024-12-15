import pandas as pd
import numpy as np
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
    'HIIT': 0,  # Mixed effect
    'Aerobic Workout': -1,
    'Hike': -1,
    'Zumba': -1,
    'Yoga': -1,
    'Swimming': -1,
    'Weights': 1,
    'Tennis': 1,
    'Sport': 1,  # Competitive sports may increase glucose
    'Workout': 0,  # General workout; effect may vary
    'Outdoor Bike': -1,
    'Walk': -1,
    'Aerobic': -1,
    'Running': -1,
    'Strength': -1,
    # Add any other activities as needed
}

# Process activity data with corrected parsing

def process_activity_data(data):
    # Identify activity columns
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    
    # For each activity column, map the activity to its impact
    for col in activity_columns:
        # Map activities to their impact
        data[col] = data[col].map(activity_impact)
    
    # Replace NaN values in activity columns with 0 (no activity)
    data[activity_columns] = data[activity_columns].fillna(0)
    
    # Extract lag times from column names
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

# Feature Engineering (simplified)
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

# Prepare data for modeling
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - exclude_cols

# Ensure both datasets have the same columns
all_columns = list(df_columns | dt_columns)

for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan
    if col not in dt.columns:
        dt[col] = np.nan

X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Impute remaining missing values (should be minimal after interpolation)
X = X.fillna(0)
X_test = X_test.fillna(0)

# Feature Selection using LightGBM
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X, y)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 50  # Reduced N
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

X_selected = X[selected_features]
X_test_selected = X_test[selected_features]

# Cross-validation with GroupKFold
groups = df['p_num']
gkf = GroupKFold(n_splits=5)

lgbm_model = LGBMRegressor(random_state=42,early_stopping_rounds=10)

rmse_scores = []
for train_index, val_index in gkf.split(X_selected, y, groups=groups):
    X_train, X_val = X_selected.iloc[train_index], X_selected.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    y_pred = lgbm_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)

print(f'Average RMSE from cross-validation: {np.mean(rmse_scores)}')

# Hyperparameter Tuning with RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 10, 15],
    'subsample': [0.6, 0.8, 1.0]
}

lgbm_random = RandomizedSearchCV(
    estimator=LGBMRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=GroupKFold(n_splits=5),
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

lgbm_random.fit(X_selected, y, groups=groups)

print(f'Best RMSE: {-lgbm_random.best_score_}')
print(f'Best Parameters: {lgbm_random.best_params_}')

# Train Final Model
best_params = lgbm_random.best_params_
lgbm_final = LGBMRegressor(**best_params, random_state=42)
lgbm_final.fit(X_selected, y)

# Make Predictions
predictions = lgbm_final.predict(X_test_selected)

# Create Submission File
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_with_activity_2.csv', index=False)
