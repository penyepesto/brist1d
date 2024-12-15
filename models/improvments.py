import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
import re
import warnings
import optuna
from optuna.integration import LightGBMPruningCallback
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

# Handle activity data (simplified for now)
activity_cols = [col for col in df.columns if col.startswith('activity')]
df[activity_cols] = df[activity_cols].notnull().astype(int)
dt[activity_cols] = dt[activity_cols].notnull().astype(int)

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

# Step 1: Aggregate data into 15-minute intervals
def aggregate_to_15_min_intervals(data):
    feature_types = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals']
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

# Step 2: Create lag features for bg
def create_lag_features(data, n_lags=4):
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    bg_cols = sorted(bg_cols, key=lambda x: int(x.split('_')[1]))
    for i in range(1, n_lags + 1):
        data[f'bg_lag_{i}'] = data[bg_cols].shift(i, axis=1).iloc[:, -1]
    return data

df = create_lag_features(df)
dt = create_lag_features(dt)

# Step 3: Rolling statistics
def create_rolling_features(data):
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    data['bg_roll_mean'] = data[bg_cols].mean(axis=1)
    data['bg_roll_std'] = data[bg_cols].std(axis=1)
    return data

df = create_rolling_features(df)
dt = create_rolling_features(dt)

# Step 4: Derivative features
def create_derivative_features(data):
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    data['bg_diff'] = data[bg_cols].diff(axis=1).iloc[:, -1]
    data['bg_pct_change'] = data[bg_cols].pct_change(axis=1).iloc[:, -1]
    return data

df = create_derivative_features(df)
dt = create_derivative_features(dt)

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

# Handle missing data using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Cross-validation with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Function to optimize hyperparameters using Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'early_stopping_rounds' : 30,
        'verbose_eval' :100,
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 10.0),
    }

    rmse_scores = []
    for train_index, val_index in tscv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Scale features
        X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_train_fold.columns)
        X_val_fold_scaled = pd.DataFrame(scaler.transform(X_val_fold), columns=X_val_fold.columns)

        train_data = lgb.Dataset(X_train_fold_scaled, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold_scaled, label=y_val_fold)

        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            #early_stopping_rounds=50,
            #verbose_eval=False,
            callbacks=[LightGBMPruningCallback(trial, 'rmse')],
        )

        y_pred = gbm.predict(X_val_fold_scaled)
        rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)

# Retrain model on full data with best hyperparameters
best_params = study.best_params
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['verbosity'] = -1

# Scale features
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

train_data = lgb.Dataset(X_scaled, label=y)

gbm = lgb.train(
    best_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data],
    #early_stopping_rounds=50,
    #verbose_eval=100,
)

# Make predictions on the test set
predictions = gbm.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_with_improvements.csv', index=False)
