import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

windows = [12, 36, 48]  # 1 hour, 3 hours, and 4 hours
df = create_rolling_features(df, windows, feature_cols)
dt = create_rolling_features(dt, windows, feature_cols)

# Step 7: Create EWMA features 
def create_ewma_features(data, feature_cols, span_values):
    for col in feature_cols:
        for span in span_values:
            ewma_col_name = f'{col}_ewma_{span}'
            data[ewma_col_name] = data.groupby('p_num')[col].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return data

span_values = [12, 24]  # Half-hour and hour EWMAs
df = create_ewma_features(df, feature_cols=['bgminus0_00'], span_values=span_values)
dt = create_ewma_features(dt, feature_cols=['bgminus0_00'], span_values=span_values)

# Step 8: Create interaction terms
df['bg_hour_interaction'] = df['bgminus0_00'] * df['hour']
dt['bg_hour_interaction'] = dt['bgminus0_00'] * dt['hour']

# Step 9: Encode 'p_num'
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

# Step 10: Handle remaining missing values

# Identify non-numeric columns in df and dt
non_numeric_cols_df = df.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric_cols_dt = dt.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove 'id' and 'time' from non-numeric columns to keep them
non_numeric_cols_df = [col for col in non_numeric_cols_df if col not in ['id', 'time']]
non_numeric_cols_dt = [col for col in non_numeric_cols_dt if col not in ['id', 'time']]

# Drop non-numeric columns if any
if non_numeric_cols_df:
    df.drop(non_numeric_cols_df, axis=1, inplace=True)
if non_numeric_cols_dt:
    dt.drop(non_numeric_cols_dt, axis=1, inplace=True)

# Now fill missing values with median in numeric columns only
numeric_cols_df = df.select_dtypes(include=[np.number]).columns
numeric_cols_dt = dt.select_dtypes(include=[np.number]).columns

df[numeric_cols_df] = df[numeric_cols_df].fillna(df[numeric_cols_df].median())
dt[numeric_cols_dt] = dt[numeric_cols_dt].fillna(dt[numeric_cols_dt].median())

# Step 11: Ensure both datasets have the same columns
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

# Step 12: Remove outliers
from scipy import stats
z_scores = np.abs(stats.zscore(y))
df = df[(z_scores < 3)]
X = df[all_columns]
y = df[target_col]

# Step 13: Target Transformation
# Apply log transformation to the target variable
y = np.log1p(y)
# Step 13: Split data into training and hold-out sets
X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
    X, y, test_size=0.1, shuffle=False
)

# Separate 'p_num_encoded' before scaling
X_train_full_pnum = X_train_full['p_num_encoded']
X_holdout_pnum = X_holdout['p_num_encoded']
X_test_pnum = X_test['p_num_encoded']

# Drop 'p_num_encoded' before scaling
X_train_full_to_scale = X_train_full.drop('p_num_encoded', axis=1)
X_holdout_to_scale = X_holdout.drop('p_num_encoded', axis=1)
X_test_to_scale = X_test.drop('p_num_encoded', axis=1)

# Step 14: Scale the data
scaler = StandardScaler()
X_train_full_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_full_to_scale),
    columns=X_train_full_to_scale.columns,
    index=X_train_full_to_scale.index
)
X_holdout_scaled = pd.DataFrame(
    scaler.transform(X_holdout_to_scale),
    columns=X_holdout_to_scale.columns,
    index=X_holdout_to_scale.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_to_scale),
    columns=X_test_to_scale.columns,
    index=X_test_to_scale.index
)

# Add 'p_num_encoded' back to the scaled data
X_train_full_scaled['p_num_encoded'] = X_train_full_pnum.values
X_holdout_scaled['p_num_encoded'] = X_holdout_pnum.values
X_test_scaled['p_num_encoded'] = X_test_pnum.values

# Step 16: Feature Selection using LightGBM feature importances
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X_train_full_scaled.drop('p_num_encoded', axis=1), y_train_full)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 300
indices = np.argsort(feature_importances)[-N:]
selected_features = [X_train_full_scaled.drop('p_num_encoded', axis=1).columns[i] for i in indices]

# Ensure 'p_num_encoded' is included
selected_features.append('p_num_encoded')

# Update datasets with selected features
X_train_full_selected = X_train_full_scaled[selected_features]
X_holdout_selected = X_holdout_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# Step 17: Custom Cross-Validation with GroupKFold
gkf = GroupKFold(n_splits=5)
groups = X_train_full_selected['p_num_encoded']


# Step 18: Hyperparameter Optimization with Optuna
def objective(trial):
    param = {
        'early_stopping_round' : 30,
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': trial.suggest_int('num_leaves', 24, 45),
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 3),
        'min_child_weight': trial.suggest_float('min_child_weight', 5, 50),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1),
    }
    
    lgtrain = lgb.Dataset(X_train_full_selected, label=y_train_full)
    cv_scores = lgb.cv(param, lgtrain, nfold=3, seed=42,
                       stratified=False, metrics=[param['metric']])
    print("cv_result keys:", cv_scores.keys())  # Inspect keys to confirm
    metric_key = f"valid {param['metric']}-mean"
    if metric_key in cv_scores:
        return -min(cv_scores[metric_key])
    else:
        raise KeyError(f"Expected metric key '{metric_key}' not found in cv_result. Available keys: {cv_scores.keys()}")


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params['learning_rate'] = 0.005
best_params['n_estimators'] = 5000
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['verbosity'] = -1
best_params['boosting_type'] = 'gbdt'
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['early_stopping_rounds'] = 30

# Step 19: Train Final LightGBM Model with Early Stopping
lgbm_final = LGBMRegressor(**best_params)

lgbm_final.fit(
    X_train_full_selected, y_train_full,
    eval_set=[(X_holdout_selected, y_holdout)],
    #early_stopping_rounds=50,
    #verbose=False
)

# Evaluate on hold-out set
y_pred_holdout = lgbm_final.predict(X_holdout_selected)
rmse_holdout = mean_squared_error(y_holdout, y_pred_holdout, squared=False)
print(f'Hold-Out RMSE (LightGBM): {rmse_holdout}')

# Step 20: Train Additional Models (XGBoost and CatBoost)
# XGBoost
xgb_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.005,
    early_stopping_rounds=30,
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
    #early_stopping_rounds=50,
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
    early_stopping_rounds=50,
    #verbose=False
)
cat_model.fit(
    X_train_full_selected, y_train_full,
    eval_set=(X_holdout_selected, y_holdout)
)

# Evaluate on hold-out set
y_pred_holdout_cat = cat_model.predict(X_holdout_selected)
rmse_holdout_cat = mean_squared_error(y_holdout, y_pred_holdout_cat, squared=False)
print(f'Hold-Out RMSE (CatBoost): {rmse_holdout_cat}')

# Step 21: Meta-Model Stacking with Ridge Regression
# Prepare stacking data
stack_train = np.column_stack([lgbm_final.predict(X_train_full_selected), xgb_model.predict(X_train_full_selected), cat_model.predict(X_train_full_selected)])
stack_holdout = np.column_stack([y_pred_holdout, y_pred_holdout_xgb, y_pred_holdout_cat])
stack_test = np.column_stack([lgbm_final.predict(X_test_selected), xgb_model.predict(X_test_selected), cat_model.predict(X_test_selected)])

# Train meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, y_train_full)

# Evaluate on hold-out set
meta_holdout_pred = meta_model.predict(stack_holdout)
rmse_meta_holdout = mean_squared_error(y_holdout, meta_holdout_pred, squared=False)
print(f'Hold-Out RMSE (Meta-Model): {rmse_meta_holdout}')

# Step 22: LSTM Model
time_steps = 12  # Ensure this matches your LSTM sequence length

# Prepare data for LSTM
def create_lstm_data(X, y, time_steps):
    X_lstm = []
    y_lstm = []
    for i in range(time_steps, len(X)):
        X_lstm.append(X.iloc[i-time_steps:i].values)
        y_lstm.append(y.iloc[i])
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    return X_lstm, y_lstm

# Prepare LSTM data for training
X_lstm_train, y_lstm_train = create_lstm_data(X_train_full_selected.reset_index(drop=True), y_train_full.reset_index(drop=True), time_steps)

# Prepare LSTM data for holdout
X_lstm_holdout, y_lstm_holdout = create_lstm_data(X_holdout_selected.reset_index(drop=True), y_holdout.reset_index(drop=True), time_steps)

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Train LSTM model
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=64, validation_data=(X_lstm_holdout, y_lstm_holdout), verbose=1)

# Generate LSTM predictions
y_pred_lstm_train = lstm_model.predict(X_lstm_train).flatten()
y_pred_lstm_holdout = lstm_model.predict(X_lstm_holdout).flatten()

# Prepare predictions from other models and align
# For training data
lgbm_pred_train = lgbm_final.predict(X_train_full_selected)[time_steps:]
xgb_pred_train = xgb_model.predict(X_train_full_selected)[time_steps:]
cat_pred_train = cat_model.predict(X_train_full_selected)[time_steps:]

# Stack predictions
stack_train = np.column_stack([lgbm_pred_train, xgb_pred_train, cat_pred_train, y_pred_lstm_train])
y_train_meta = y_train_full.reset_index(drop=True)[time_steps:]

# For holdout data
y_pred_holdout_lgbm = lgbm_final.predict(X_holdout_selected)[time_steps:]
y_pred_holdout_xgb = xgb_model.predict(X_holdout_selected)[time_steps:]
y_pred_holdout_cat = cat_model.predict(X_holdout_selected)[time_steps:]

# Stack predictions
stack_holdout = np.column_stack([y_pred_holdout_lgbm, y_pred_holdout_xgb, y_pred_holdout_cat, y_pred_lstm_holdout])
y_holdout_meta = y_holdout.reset_index(drop=True)[time_steps:]

# Ensure all arrays have the same length
assert stack_train.shape[0] == y_train_meta.shape[0]
assert stack_holdout.shape[0] == y_holdout_meta.shape[0]

# Train meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, y_train_meta)

# Evaluate on hold-out set
meta_holdout_pred = meta_model.predict(stack_holdout)
rmse_meta_holdout = mean_squared_error(y_holdout_meta, meta_holdout_pred, squared=False)
print(f'Hold-Out RMSE (Meta-Model): {rmse_meta_holdout}')

# Prepare LSTM data for test set
X_lstm_test = []
for i in range(time_steps, len(X_test_selected)):
    X_lstm_test.append(X_test_selected.iloc[i-time_steps:i].values)
X_lstm_test = np.array(X_lstm_test)

# Generate LSTM predictions on test set
y_pred_lstm_test = lstm_model.predict(X_lstm_test).flatten()

# Adjust other models' predictions
lgbm_pred_test = lgbm_final.predict(X_test_selected)[time_steps:]
xgb_pred_test = xgb_model.predict(X_test_selected)[time_steps:]
cat_pred_test = cat_model.predict(X_test_selected)[time_steps:]

# Stack test predictions
stack_test = np.column_stack([lgbm_pred_test, xgb_pred_test, cat_pred_test, y_pred_lstm_test])

# Generate final predictions using meta-model
final_predictions = meta_model.predict(stack_test)
final_predictions = np.expm1(final_predictions)

# Predictions for the first 'time_steps' samples
first_predictions = lgbm_final.predict(X_test_selected[:time_steps])
first_predictions = np.expm1(first_predictions)

# Combine all predictions
full_predictions = np.concatenate([first_predictions, final_predictions])

# Ensure the length matches the test set
assert len(full_predictions) == len(X_test_selected)

# Create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': full_predictions})
submission.to_csv('submission_lstm_2.csv', index=False)