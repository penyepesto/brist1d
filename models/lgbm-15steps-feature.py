import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import re
import warnings
import optuna
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Step 1: Data Preprocessing
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
    
    # Sum activity impact over the last hour
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

# Handle missing values using interpolation
def interpolate_missing_values(data):
    data.sort_values(['p_num', 'time'], inplace=True)
    data = data.groupby('p_num').apply(lambda group: group.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill'))
    return data.reset_index(drop=True)

df = interpolate_missing_values(df)
dt = interpolate_missing_values(dt)

# Step 4: Create additional features based on relationships
def create_additional_features(data):
    # Insulin sensitivity: insulin / bg (avoid division by zero)
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    for ins_col, bg_col in zip(sorted(insulin_cols), sorted(bg_cols)):
        data[f'insulin_sensitivity_{ins_col[-5:]}'] = data[ins_col] / (data[bg_col] + 1e-5)
    # Interaction terms
    # Activity impact * insulin
    activity_cols = [col for col in data.columns if col.startswith('activity_')]
    for act_col, ins_col in zip(sorted(activity_cols), sorted(insulin_cols)):
        data[f'activity_insulin_{act_col[-5:]}'] = data[act_col] * data[ins_col]
    # Difference in bg over time
    sorted_bg_cols = sorted(bg_cols)
    for i in range(len(sorted_bg_cols)-1):
        data[f'bg_diff_{i}'] = data[sorted_bg_cols[i]] - data[sorted_bg_cols[i+1]]
    return data

df = create_additional_features(df)
dt = create_additional_features(dt)

# Step 5: Rolling Window Features
def add_rolling_features(data, window_sizes=[3,6,12]):
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    carbs_cols = [col for col in data.columns if col.startswith('carbs_')]
    hr_cols = [col for col in data.columns if col.startswith('hr_')]
    
    for window in window_sizes:
        # Rolling mean
        data[f'bg_roll_mean_{window}'] = data[bg_cols].rolling(window, axis=1).mean().max(axis=1)
        data[f'insulin_roll_mean_{window}'] = data[insulin_cols].rolling(window, axis=1).mean().max(axis=1)
        data[f'carbs_roll_mean_{window}'] = data[carbs_cols].rolling(window, axis=1).mean().max(axis=1)
        data[f'hr_roll_mean_{window}'] = data[hr_cols].rolling(window, axis=1).mean().max(axis=1)
        # Rolling std
        data[f'bg_roll_std_{window}'] = data[bg_cols].rolling(window, axis=1).std().max(axis=1)
        data[f'insulin_roll_std_{window}'] = data[insulin_cols].rolling(window, axis=1).std().max(axis=1)
        data[f'carbs_roll_std_{window}'] = data[carbs_cols].rolling(window, axis=1).std().max(axis=1)
        data[f'hr_roll_std_{window}'] = data[hr_cols].rolling(window, axis=1).std().max(axis=1)
    return data

df = add_rolling_features(df)
dt = add_rolling_features(dt)

# Step 6: Exponential Moving Averages
def add_ema_features(data, span=3):
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    data['bg_ema'] = data[bg_cols].ewm(span=span, axis=1).mean().max(axis=1)
    return data

df = add_ema_features(df)
dt = add_ema_features(dt)

# Step 7: Feature Scaling
scaler = RobustScaler()
features_to_scale = [col for col in df.columns if col not in ['id', 'time', 'p_num', target_col]]
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
dt[features_to_scale] = scaler.transform(dt[features_to_scale])

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

# Impute remaining missing values
X = X.fillna(0)
X_test = X_test.fillna(0)

# Feature Selection based on importance
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X, y)
feature_importances = temp_lgbm.feature_importances_

N = 1000  # Increased number of features
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

X = X[selected_features]
X_test = X_test[selected_features]

# Cross-validation with GroupKFold and early stopping
groups = df['p_num']
gkf = GroupKFold(n_splits=5)

best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.005,
    'max_depth': 15,
    'n_estimators': 2000,
    'num_leaves': 31,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'subsample': 1.0,
    'random_state': 42,
    'verbose': -1,
    'early_stopping_rounds': 50
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
submission.to_csv('submission_with_aggregated_features44.csv', index=False)
print("finally")

""" 
# Step 8: Initialize Models Without Early Stopping
# Remove early_stopping_rounds from base estimators

lgbm_params = {    
    'n_estimators': 1000,
    'learning_rate': 0.005,
    'num_leaves': 31,
    'max_depth': 15,
    'reg_alpha':10.0,
    'reg_lambda':10.0,
    'subsample':1.0,
    'colsample_bytree':1.0,
    'random_state':42,
    'n_jobs':-1,
    'early_stopping_rounds' : 10
    }
lgbm_model = LGBMRegressor(**lgbm_params)

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 10,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'logging_level': 'Silent',
    'early_stopping_rounds': 10  # Optional: You can keep early stopping for CatBoost as it handles it internally
}

cat_model = CatBoostRegressor(**cat_params)

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds' : 10
}

xgb_model = XGBRegressor(**xgb_params)

# Step 9: Ensembling with Stacking Regressor
estimators = [
    ('lgbm', lgbm_model),
    ('cat', cat_model),
    ('xgb', xgb_model)
]

# Initialize Stacking Regressor without early stopping in base estimators
stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LGBMRegressor(
        learning_rate=0.01,
        n_estimators=1000,
        num_leaves=31,
        max_depth=10,
        reg_alpha=10.0,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=-1
    ),
    n_jobs=-1,
    passthrough=True  # Optional: Pass original features to the final estimator
)

# Step 10: Cross-Validation and Model Training
rmse_scores = []
folds = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y, groups=df['p_num'])):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    stacking_reg.fit(X_train, y_train)
    
    y_pred = stacking_reg.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    print(f'Fold {fold+1} RMSE: {rmse}')

print(f'Average RMSE from cross-validation: {np.mean(rmse_scores)}')

# Step 11: Train Final Model on Full Data
stacking_reg.fit(X, y)

# Step 12: Make Predictions on the Test Set
predictions = stacking_reg.predict(X_test)

# Step 13: Create Submission File
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('11submission_stacked_model.csv', index=False)
print("finally") """