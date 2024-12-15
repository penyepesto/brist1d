# Install necessary libraries


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from boruta import BorutaPy

# Read the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Convert 'time' to datetime
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
dt['time'] = pd.to_datetime(dt['time'], format='%H:%M:%S')

# Extract time-based features
def extract_time_features(data):
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['second'] = data['time'].dt.second
    # Cyclical transformation
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    return data

df = extract_time_features(df)
dt = extract_time_features(dt)

# Fill missing values
bg_columns = [col for col in df.columns if col.startswith('bg-')]

def fill_missing_values(data):
    # Fill 'bg' columns
    data[bg_columns] = data[bg_columns].fillna(method='ffill').fillna(method='bfill')
    # Fill 'hr', 'steps', 'cals'
    for prefix in ['hr', 'steps', 'cals']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        data[cols] = data[cols].fillna(method='ffill').fillna(method='bfill')
    # Fill 'insulin' and 'carbs' with 0
    for prefix in ['insulin', 'carbs']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        data[cols] = data[cols].fillna(0)
    # Drop 'activity' columns
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    data.drop(activity_columns, axis=1, inplace=True)
    return data

df = fill_missing_values(df)
dt = fill_missing_values(dt)

# Feature Engineering function
def feature_engineering(data):
    data = data.copy()
    data = data.sort_values(by=['p_num', 'time'])
    bg_columns = [col for col in data.columns if col.startswith('bg-')]
    lag_steps = [1, 2, 3, 4, 5, 6]  # Lag steps (5-minute intervals)
    rolling_windows = [6, 12, 24, 36]  # Windows in 5-minute intervals

    for col in bg_columns:
        # Lagged features
        for lag in lag_steps:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
        # Rolling features
        for window in rolling_windows:
            data[f'{col}_rolling_mean_{window}'] = data.groupby('p_num')[col].rolling(window).mean().reset_index(level=0, drop=True)
            data[f'{col}_rolling_std_{window}'] = data.groupby('p_num')[col].rolling(window).std().reset_index(level=0, drop=True)
            data[f'{col}_ewm_mean_{window}'] = data.groupby('p_num')[col].ewm(span=window).mean().reset_index(level=0, drop=True)

    # Interaction terms
    data['bg_hr_interaction'] = data['bg-0:00'] * data['hr-0:00']
    data['bg_steps_interaction'] = data['bg-0:00'] * data['steps-0:00']
    data['insulin_carbs_interaction'] = data['insulin-0:00'] * data['carbs-0:00']

    # Cyclical features for heart rate and steps (if applicable)
    # Additional features can be added here

    # Fill any remaining NaNs
    data = data.fillna(method='bfill').fillna(method='ffill')
    return data

# Apply feature engineering
df = feature_engineering(df)
dt = feature_engineering(dt)

# Ensure both datasets have the same columns
target_col = 'bg+1:00'
exclude_cols = {'id', 'time', target_col}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - {'id', 'time'}

# Get the union of columns
all_columns = list(df_columns | dt_columns)

# Ensure both datasets have the same columns
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan
    if col not in dt.columns:
        dt[col] = np.nan

# Ensure the columns are in the same order
all_columns = sorted(all_columns)
X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Drop 'p_num' as participants in test are unseen
if 'p_num' in X.columns:
    X = X.drop(columns=['p_num'])
    X_test = X_test.drop(columns=['p_num'])

# Impute missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Feature Selection using Boruta
# Convert arrays back to DataFrame for Boruta
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
y_series = pd.Series(y.values)

# Initialize Boruta
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
boruta_selector.fit(X_scaled_df.values, y_series.values)

# Selected features
selected_features = X_scaled_df.columns[boruta_selector.support_].tolist()
X_selected = X_scaled_df[selected_features]
X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[selected_features]

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Define models with initial parameters
lgbm = LGBMRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
cat = CatBoostRegressor(random_state=42, verbose=0)
ridge = Ridge()

# Hyperparameter Tuning using Bayesian Optimization
def optimize_lgbm(X, y):
    def lgbm_cv(max_depth, num_leaves, learning_rate, n_estimators, subsample):
        params = {
            'max_depth': int(max_depth),
            'num_leaves': int(num_leaves),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'subsample': subsample,
            'random_state': 42
        }
        model = LGBMRegressor(**params)
        rmse = -np.mean(cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error'))
        return -rmse

    lgbm_bo = BayesianOptimization(
        lgbm_cv,
        {
            'max_depth': (3, 10),
            'num_leaves': (20, 50),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (100, 1000),
            'subsample': (0.5, 1)
        },
        random_state=42
    )
    lgbm_bo.maximize(init_points=5, n_iter=25)
    return lgbm_bo.max['params']

def optimize_xgb(X, y):
    def xgb_cv(max_depth, learning_rate, n_estimators, colsample_bytree, subsample):
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'random_state': 42,
            'tree_method': 'hist'
        }
        model = XGBRegressor(**params)
        rmse = -np.mean(cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error'))
        return -rmse

    xgb_bo = BayesianOptimization(
        xgb_cv,
        {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (100, 1000),
            'colsample_bytree': (0.5, 1),
            'subsample': (0.5, 1)
        },
        random_state=42
    )
    xgb_bo.maximize(init_points=5, n_iter=25)
    return xgb_bo.max['params']

def optimize_cat(X, y):
    def cat_cv(depth, learning_rate, n_estimators, l2_leaf_reg):
        params = {
            'depth': int(depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'l2_leaf_reg': l2_leaf_reg,
            'random_state': 42
        }
        model = CatBoostRegressor(**params, verbose=0)
        rmse = -np.mean(cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error'))
        return -rmse

    cat_bo = BayesianOptimization(
        cat_cv,
        {
            'depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (100, 1000),
            'l2_leaf_reg': (1, 10)
        },
        random_state=42
    )
    cat_bo.maximize(init_points=5, n_iter=25)
    return cat_bo.max['params']

# Optimize models
lgbm_params = optimize_lgbm(X_selected.values, y_series.values)
xgb_params = optimize_xgb(X_selected.values, y_series.values)
cat_params = optimize_cat(X_selected.values, y_series.values)

# Train models with optimized parameters
lgbm_model = LGBMRegressor(
    max_depth=int(lgbm_params['max_depth']),
    num_leaves=int(lgbm_params['num_leaves']),
    learning_rate=lgbm_params['learning_rate'],
    n_estimators=int(lgbm_params['n_estimators']),
    subsample=lgbm_params['subsample'],
    random_state=42
)
lgbm_model.fit(X_selected, y_series)

xgb_model = XGBRegressor(
    max_depth=int(xgb_params['max_depth']),
    learning_rate=xgb_params['learning_rate'],
    n_estimators=int(xgb_params['n_estimators']),
    colsample_bytree=xgb_params['colsample_bytree'],
    subsample=xgb_params['subsample'],
    random_state=42,
    tree_method='hist'
)
xgb_model.fit(X_selected, y_series)

cat_model = CatBoostRegressor(
    depth=int(cat_params['depth']),
    learning_rate=cat_params['learning_rate'],
    n_estimators=int(cat_params['n_estimators']),
    l2_leaf_reg=cat_params['l2_leaf_reg'],
    random_state=42,
    verbose=0
)
cat_model.fit(X_selected, y_series)

# Ensemble using Stacking Regressor
estimators = [
    ('lgbm', lgbm_model),
    ('xgb', xgb_model),
    ('cat', cat_model)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=tscv
)
stacking_model.fit(X_selected, y_series)

# Make predictions
ensemble_pred = stacking_model.predict(X_test_selected)

# Create submission
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': ensemble_pred})
submission.to_csv('submission.csv', index=False)