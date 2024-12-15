import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Step 2: Convert 'time' to datetime and extract time-based features
def extract_time_features(data):
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data['day_of_week'] = data['time'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5,6]).astype(int)
    # Cyclical transformation
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 60)
    # Cyclical encoding for day of week
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
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
target_col = 'bgplus1_00'  # Updated target name after cleaning


# Step 4: Handle missing values using forward/backward fill within groups
def fill_missing_values_grouped(data):
    data = data.groupby('p_num').apply(lambda group: group.fillna(method='ffill').fillna(method='bfill'))
    data = data.reset_index(drop=True)  # Reset index to fix 'p_num' ambiguity
    return data

df = fill_missing_values_grouped(df)
dt = fill_missing_values_grouped(dt)

# Step 5: Create lag features
def create_lag_features(data, lags, feature_cols):
    for col in feature_cols:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

lags = range(1, 13)  # Lag features for past 1 hour (assuming 5-minute intervals)
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = create_lag_features(df, lags, feature_cols)
dt = create_lag_features(dt, lags, feature_cols)

# Step 6: Create rolling statistics
def create_rolling_features(data, windows, feature_cols):
    for col in feature_cols:
        for window in windows:
            data[f'{col}_rolling_mean_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            data[f'{col}_rolling_std_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
    return data

windows = [12, 24, 36]  # Rolling windows of 1, 2, and 3 hours
df = create_rolling_features(df, windows, feature_cols)
dt = create_rolling_features(dt, windows, feature_cols)

# Step 7: Create interaction features
df['hour_bg_interaction'] = df['hour'] * df['bgminus0_00']
dt['hour_bg_interaction'] = dt['hour'] * dt['bgminus0_00']

# Step 8: Drop 'activity' columns if they exist
activity_columns = [col for col in df.columns if col.startswith('activity')]
df.drop(columns=activity_columns, inplace=True)

activity_columns_dt = [col for col in dt.columns if col.startswith('activity')]
dt.drop(columns=activity_columns_dt, inplace=True)

# Proceed with previous steps
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - {'id', 'time', 'p_num'}

# Step 9: Ensure both datasets have the same columns
all_columns = list(df_columns | dt_columns)

# Ensure both datasets have the same columns and order
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan
    if col not in dt.columns:
        dt[col] = np.nan
all_columns = sorted(all_columns)

# Step 11: Handle remaining missing values by filling with median
def fill_remaining_missing_values(data):
    # Identify numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    # Fill missing values in numeric columns with median
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # For non-numeric columns, drop them (excluding 'id', 'time', 'p_num' if needed)
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col not in ['id', 'time', 'p_num']]
    data.drop(columns=non_numeric_cols, inplace=True)

    return data

df = fill_remaining_missing_values(df)
dt = fill_remaining_missing_values(dt)

# Update all_columns after dropping non-numeric columns
all_columns = [col for col in all_columns if col in df.columns]

# Step 12: Remove or cap outliers in the target variable
upper_limit = df[target_col].quantile(0.99)
df[target_col] = df[target_col].clip(upper=upper_limit)

# Step 13: Transform the target variable
y = df[target_col]
y_transformed = np.log1p(y)

# Step 14: Prepare feature matrix X and test data
X = df[all_columns]
X_test = dt[all_columns]

# Step 15: Scale the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Step 16: Feature selection using LightGBM feature importances
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X_scaled, y_transformed)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 200  # Increase N to include more features
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

# Update X and X_test with selected features
X_selected = X_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# Step 17: Use Recursive Feature Elimination (RFE)
rfe = RFECV(estimator=LGBMRegressor(random_state=42), step=10, cv=TimeSeriesSplit(n_splits=5), scoring='neg_root_mean_squared_error', n_jobs=-1)
rfe.fit(X_selected, y_transformed)
selected_features_rfe = X_selected.columns[rfe.support_]

# Update X and X_test with RFE selected features
X_selected = X_selected[selected_features_rfe]
X_test_selected = X_test_selected[selected_features_rfe]

""" # Step 18: Advanced Hyperparameter Tuning with Bayesian Optimization


params = {
    'num_leaves': Integer(20, 100),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.001, 0.1, 'log-uniform'),
    'n_estimators': Integer(100, 2000),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'reg_alpha': Real(0, 10),
    'reg_lambda': Real(0, 10)
}

lgbm = LGBMRegressor(random_state=42)

opt = BayesSearchCV(
    estimator=lgbm,
    search_spaces=params,
    n_iter=32,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

opt.fit(X_selected, y_transformed)
print(f'Best RMSE: {-opt.best_score_}')
print(f'Best Parameters: {opt.best_params_}')

# Step 19: Adjust model training with early stopping and increased estimators
best_params = opt.best_params_
best_params['learning_rate'] = best_params.get('learning_rate', 0.005)
best_params['n_estimators'] = best_params.get('n_estimators', 5000)
best_params['random_state'] = 42
best_params['early_stopping_rounds'] = 10

lgbm_final = LGBMRegressor(**best_params)
 """
# Step 18: Define best_params directly bc i run it before

best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.0036326616822094098,
    'max_depth': 15,
    'n_estimators': 1619,
    'num_leaves': 31, #20 
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'subsample': 1.0,
    'early_stopping_rounds' : 10,
    'random_state': 42
}

# Optionally adjust 'n_estimators' and 'learning_rate' for better performance
#best_params['n_estimators'] = 2000  # Increased from 1619
#best_params['learning_rate'] = 0.005  # Adjusted as needed
#best_params['early_stopping_rounds'] = 10  

# Step 19: Initialize the model
#lgbm_final = LGBMRegressor(**best_params)
"""
# Step 20: Cross-validation strategy using GroupKFold
groups = df['p_num']
gkf = GroupKFold(n_splits=5)

 rmse_scores = []
for train_index, val_index in gkf.split(X_selected, y_transformed, groups):
    X_train, X_val = X_selected.iloc[train_index], X_selected.iloc[val_index]
    y_train, y_val = y_transformed.iloc[train_index], y_transformed.iloc[val_index]

    lgbm_final.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        #early_stopping_rounds=50,
        #verbose=False
    )
    y_pred = lgbm_final.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)

print(f'Average RMSE from cross-validation: {np.mean(rmse_scores)}')
""" 
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_selected, y_transformed, test_size=0.1, random_state=42
)
# Step 21: Train final LightGBM model on full data
lgbm_final = LGBMRegressor(**best_params)
lgbm_final.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val, y_val)],
    #early_stopping_rounds=50,
    #verbose=False
)

# Step 22: Train additional models (XGBoost and CatBoost)
# XGBoost
xgb_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val, y_val)],
)
# CatBoost
cat_model = CatBoostRegressor(
    iterations=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    depth=best_params['max_depth'],
    random_seed=42,
    early_stopping_rounds=50,
    verbose=0
)
cat_model.fit(
    X_train_full, y_train_full,
    eval_set=(X_val, y_val),
)
# Step 23: Ensemble predictions
pred_lgbm = lgbm_final.predict(X_test_selected)
pred_xgb = xgb_model.predict(X_test_selected)
pred_cat = cat_model.predict(X_test_selected)

# Since we transformed the target variable, inverse transform predictions
pred_lgbm = np.expm1(pred_lgbm)
pred_xgb = np.expm1(pred_xgb)
pred_cat = np.expm1(pred_cat)

# Combine predictions (simple average or weighted average)
# If you have validation scores, you can adjust weights accordingly
predictions = (pred_lgbm + pred_xgb + pred_cat) / 3

# Step 24: Make predictions and create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_ensemble.csv', index=False)

# Step 25: Analyze feature importances
feature_imp = pd.DataFrame({
    'Feature': selected_features_rfe,
    'Importance': lgbm_final.feature_importances_
})
feature_imp.sort_values('Importance', ascending=False, inplace=True)

plt.figure(figsize=(12,6))
plt.barh(feature_imp['Feature'][:20], feature_imp['Importance'][:20])
plt.gca().invert_yaxis()
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
