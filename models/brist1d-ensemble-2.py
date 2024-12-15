# Import necessary libraries
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Step 1: Read the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Step 2: Convert 'time' to datetime and extract time-based features
def extract_time_features(data):
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['day_of_week'] = data['time'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    # Cyclical encoding
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 60)
    return data

df = extract_time_features(df)
dt = extract_time_features(dt)

# Step 3: Clean column names to remove special characters
def clean_column_names(data):
    data.columns = data.columns.str.replace(':', '_', regex=False)
    data.columns = data.columns.str.replace('+', 'plus', regex=False)
    data.columns = data.columns.str.replace('-', 'minus', regex=False)
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return data

df = clean_column_names(df)
dt = clean_column_names(dt)

# Update target column name after cleaning
target_col = 'bgplus1_00'

# Step 4: Fill missing values strategically
def fill_missing_values(data):
    # Define the columns for blood glucose data, excluding the target column
    bg_columns = [col for col in data.columns if (col.startswith('bgminus') or col.startswith('bgplus')) and col != target_col]
    
    # Forward and backward fill on each participant's data for blood glucose columns
    if bg_columns:
        data[bg_columns] = data.groupby('p_num')[bg_columns].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    
    # Fill missing values for other time-series columns with forward and backward fill
    for prefix in ['hr', 'steps', 'cals']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        if cols:
            data[cols] = data.groupby('p_num')[cols].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    
    # Fill missing values for insulin and carbs columns with zero
    for prefix in ['insulin', 'carbs']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        if cols:
            data[cols] = data[cols].fillna(0)
    
    # Drop activity columns if they exist
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    data.drop(activity_columns, axis=1, inplace=True)
    
    return data

df = fill_missing_values(df)
dt = fill_missing_values(dt)

# Step 5: Create lag features
def create_lag_features(data, lag_steps, feature_cols):
    for col in feature_cols:
        for lag in lag_steps:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

lag_steps = range(1, 13)  # Up to 1 hour lag (12 * 5-minute intervals)
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = create_lag_features(df, lag_steps, feature_cols)
dt = create_lag_features(dt, lag_steps, feature_cols)

# Step 6: Create rolling statistics
def create_rolling_features(data, windows, feature_cols):
    for col in feature_cols:
        for window in windows:
            data[f'{col}_rolling_mean_{window}'] = data.groupby('p_num')[col].rolling(window).mean().reset_index(level=0, drop=True)
            data[f'{col}_rolling_std_{window}'] = data.groupby('p_num')[col].rolling(window).std().reset_index(level=0, drop=True)
    return data

windows = [12, 36]  # 1 hour and 3 hours
df = create_rolling_features(df, windows, feature_cols)
dt = create_rolling_features(dt, windows, feature_cols)

# Step 7: Create interaction terms
df['insulin_carbs_interaction'] = df['insulinminus0_00'] * df['carbsminus0_00']
dt['insulin_carbs_interaction'] = dt['insulinminus0_00'] * dt['carbsminus0_00']

# Step 8: Encode 'p_num'
from sklearn.preprocessing import LabelEncoder

# Combine 'p_num' from both datasets to ensure consistent encoding
combined_p_num = pd.concat([df['p_num'], dt['p_num']], axis=0)

# Fit LabelEncoder
le = LabelEncoder()
le.fit(combined_p_num)

# Transform 'p_num' in both datasets
df['p_num_encoded'] = le.transform(df['p_num'])
dt['p_num_encoded'] = le.transform(dt['p_num'])

# Drop the original 'p_num' column
df.drop('p_num', axis=1, inplace=True)
dt.drop('p_num', axis=1, inplace=True)

# **Step 9: Handle remaining missing values**

# **Modify this step to exclude 'id' from being dropped**

# Identify non-numeric columns, excluding 'id'
non_numeric_cols_df = df.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric_cols_dt = dt.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove 'id' and 'time' from non-numeric columns to keep them
non_numeric_cols_df = [col for col in non_numeric_cols_df if col not in ['id', 'time']]
non_numeric_cols_dt = [col for col in non_numeric_cols_dt if col not in ['id', 'time']]

# Drop non-numeric columns if any
df.drop(non_numeric_cols_df, axis=1, inplace=True)
dt.drop(non_numeric_cols_dt, axis=1, inplace=True)

# Now fill missing values with median
numeric_cols_df = df.select_dtypes(include=[np.number]).columns
numeric_cols_dt = dt.select_dtypes(include=[np.number]).columns

df[numeric_cols_df] = df[numeric_cols_df].fillna(df[numeric_cols_df].median())
dt[numeric_cols_dt] = dt[numeric_cols_dt].fillna(dt[numeric_cols_dt].median())

# **Ensure 'id' is preserved in 'dt'**
# 'id' should still be present in 'dt' at this point

# Step 10: Ensure both datasets have the same columns
exclude_cols = {'id', 'time', target_col}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - {'id', 'time'}

# Get the union of columns
all_columns = list(df_columns | dt_columns)

# Ensure both datasets have the same columns and order
for col in all_columns:
    if col not in df.columns:
        df[col] = 0
    if col not in dt.columns:
        dt[col] = 0
all_columns = sorted(all_columns)
X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Step 11: Remove outliers
from scipy import stats
z_scores = np.abs(stats.zscore(y))
df = df[(z_scores < 3)]
X = df[all_columns]
y = df[target_col]

# Step 12: Split data into training and hold-out sets
X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y, test_size=0.1, shuffle=False)

# Step 13: Scale the data
scaler = StandardScaler()
X_train_full_scaled = pd.DataFrame(scaler.fit_transform(X_train_full), columns=X_train_full.columns)
X_holdout_scaled = pd.DataFrame(scaler.transform(X_holdout), columns=X_holdout.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Step 14: Feature Selection using LightGBM feature importances
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X_train_full_scaled, y_train_full)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 200  # Increased N
indices = np.argsort(feature_importances)[-N:]
selected_features = [X_train_full_scaled.columns[i] for i in indices]

# Update datasets with selected features
X_train_full_selected = X_train_full_scaled[selected_features]
X_holdout_selected = X_holdout_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# Step 14: Advanced Hyperparameter Tuning with Bayesian Optimization
def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth,
             lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_iterations': 1000,
        'learning_rate': 0.05,
        'early_stopping_round': 50,
        'num_leaves': int(round(num_leaves)),
        'feature_fraction': max(min(feature_fraction, 1), 0),
        'bagging_fraction': max(min(bagging_fraction, 1), 0),
        'max_depth': int(round(max_depth)),
        'lambda_l1': max(lambda_l1, 0),
        'lambda_l2': max(lambda_l2, 0),
        'min_split_gain': min_split_gain,
        'min_child_weight': min_child_weight,
        'verbosity': -1,
        'n_jobs': -1
    }
    lgtrain = lgb.Dataset(X_train_full_selected, label=y_train_full)
    cv_result = lgb.cv(params, lgtrain, nfold=3, seed=42,
                       stratified=False, metrics=[params['metric']])
    print("cv_result keys:", cv_result.keys())  # Inspect keys to confirm
    metric_key = f"valid {params['metric']}-mean"
    if metric_key in cv_result:
        return -min(cv_result[metric_key])
    else:
        raise KeyError(f"Expected metric key '{metric_key}' not found in cv_result. Available keys: {cv_result.keys()}")


lgbBO = BayesianOptimization(lgb_eval, {
    'num_leaves': (24, 45),
    'feature_fraction': (0.1, 0.9),
    'bagging_fraction': (0.8, 1),
    'max_depth': (5, 9),
    'lambda_l1': (0, 5),
    'lambda_l2': (0, 3),
    'min_split_gain': (0.001, 0.1),
    'min_child_weight': (5, 50)
}, random_state=42)

lgbBO.maximize(init_points=5, n_iter=25)

# Extract best parameters
best_params = lgbBO.max['params']
best_params['num_leaves'] = int(round(best_params['num_leaves']))
best_params['max_depth'] = int(round(best_params['max_depth']))

# Step 16: Train Final LightGBM Model
lgbm_final = LGBMRegressor(
    n_estimators=5000,  # Increased to compensate for smaller learning rate
    learning_rate=0.005,
    **best_params,
    random_state=42,
    n_jobs=-1
)

lgbm_final.fit(
    X_train_full_selected, y_train_full,
    eval_set=[(X_holdout_selected, y_holdout)],
    #early_stopping_rounds=100,
    #verbose=False
)

# Evaluate on hold-out set
y_pred_holdout = lgbm_final.predict(X_holdout_selected)
rmse_holdout = mean_squared_error(y_holdout, y_pred_holdout, squared=False)
print(f'Hold-Out RMSE (LightGBM): {rmse_holdout}')

# Step 17: Train Additional Models (XGBoost and CatBoost)
# XGBoost
xgb_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=best_params['max_depth'],
    subsample=best_params['bagging_fraction'],
    colsample_bytree=best_params['feature_fraction'],
    reg_alpha=best_params['lambda_l1'],
    reg_lambda=best_params['lambda_l2'],
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train_full_selected, y_train_full,
    #early_stopping_rounds=100,
    eval_set=[(X_holdout_selected, y_holdout)],
    #verbose=False
)

# Evaluate on hold-out set
y_pred_holdout_xgb = xgb_model.predict(X_holdout_selected)
rmse_holdout_xgb = mean_squared_error(y_holdout, y_pred_holdout_xgb, squared=False)
print(f'Hold-Out RMSE (XGBoost): {rmse_holdout_xgb}')

# CatBoost
cat_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.005,
    depth=best_params['max_depth'],
    l2_leaf_reg=best_params['lambda_l2'],
    random_seed=42,
    eval_metric='RMSE',
    early_stopping_rounds=100,
    verbose=False
)
cat_model.fit(
    X_train_full_selected, y_train_full,
    eval_set=(X_holdout_selected, y_holdout)
)

# Evaluate on hold-out set
y_pred_holdout_cat = cat_model.predict(X_holdout_selected)
rmse_holdout_cat = mean_squared_error(y_holdout, y_pred_holdout_cat, squared=False)
print(f'Hold-Out RMSE (CatBoost): {rmse_holdout_cat}')

# Step 18: Ensemble Predictions
pred_lgbm = lgbm_final.predict(X_test_selected)
pred_xgb = xgb_model.predict(X_test_selected)
pred_cat = cat_model.predict(X_test_selected)

# Weighted average of predictions
predictions = (pred_lgbm * 0.4) + (pred_xgb * 0.3) + (pred_cat * 0.3)

# **Step 19: Create Submission File**

# Use 'dt' DataFrame to get the 'id' column
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_ensemble_2.csv', index=False)

# Step 20: Feature Importance Analysis
lgb.plot_importance(lgbm_final, max_num_features=20)
plt.title('LightGBM Feature Importances')
plt.show()
