import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb 
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import optuna
from optuna.integration import LightGBMPruningCallback
import warnings
warnings.filterwarnings('ignore')
import re 
# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Function to clean column names
def clean_column_names(data):
    data.columns = data.columns.str.replace(':', '_', regex=False)
    data.columns = data.columns.str.replace('+', 'plus', regex=False)
    data.columns = data.columns.str.replace('-', 'minus', regex=False)
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return data

train = clean_column_names(train)
test = clean_column_names(test)

# Update target column name
target_col = 'bgplus1_00'

# Extract time features
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

train = extract_time_features(train)
test = extract_time_features(test)

# Handle activity data
activity_cols = [col for col in train.columns if col.startswith('activity')]
activity_list = train[activity_cols].stack().unique()
activity_mapping = {activity: idx for idx, activity in enumerate(activity_list)}
activity_mapping[np.nan] = -1  # For missing activities

def process_activity(data):
    for col in activity_cols:
        data[col] = data[col].map(activity_mapping)
    return data

train = process_activity(train)
test = process_activity(test)

# Advanced Imputation using IterativeImputer
def advanced_imputation(train_data, test_data):
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), max_iter=10, random_state=42)
    combined = pd.concat([train_data, test_data], sort=False).reset_index(drop=True)
    imputed = imputer.fit_transform(combined)
    imputed_df = pd.DataFrame(imputed, columns=combined.columns)
    train_imputed = imputed_df.iloc[:len(train_data)]
    test_imputed = imputed_df.iloc[len(train_data):].reset_index(drop=True)
    return train_imputed, test_imputed

# Prepare data for imputation
features = [col for col in train.columns if col not in ['id', 'time', target_col]]
train_impute = train[features]
test_impute = test[features]

train_imputed, test_imputed = advanced_imputation(train_impute, test_impute)

# Add target back to train_imputed
train_imputed[target_col] = train[target_col]

# Feature Engineering: Interaction Features
def create_interaction_features(data):
    # Example interaction: insulin * carbs
    insulin_cols = [col for col in data.columns if 'insulinminus' in col]
    carbs_cols = [col for col in data.columns if 'carbsminus' in col]
    for i in range(len(insulin_cols)):
        data[f'insulin_carbs_{i}'] = data[insulin_cols[i]] * data[carbs_cols[i]]
    return data

train_imputed = create_interaction_features(train_imputed)
test_imputed = create_interaction_features(test_imputed)

# Lag Features
def create_lag_features(data, lag=6):
    bg_cols = [col for col in data.columns if 'bgminus' in col]
    bg_cols_sorted = sorted(bg_cols, key=lambda x: int(re.findall(r'\d+', x)[1]) * 60 + int(re.findall(r'\d+', x)[2]))
    for i in range(1, lag + 1):
        data[f'bg_lag_{i}'] = data[bg_cols_sorted].shift(i, axis=1).iloc[:, -1]
    return data

train_imputed = create_lag_features(train_imputed)
test_imputed = create_lag_features(test_imputed)

# Rolling Statistics
def create_rolling_features(data):
    bg_cols = [col for col in data.columns if 'bgminus' in col]
    data['bg_mean'] = data[bg_cols].mean(axis=1)
    data['bg_std'] = data[bg_cols].std(axis=1)
    data['bg_max'] = data[bg_cols].max(axis=1)
    data['bg_min'] = data[bg_cols].min(axis=1)
    return data

train_imputed = create_rolling_features(train_imputed)
test_imputed = create_rolling_features(test_imputed)

# Participant Encoding
def encode_participant(data, train_data):
    participant_mapping = {p: idx for idx, p in enumerate(train_data['p_num'].unique())}
    data['p_num_encoded'] = data['p_num'].map(participant_mapping)
    data['p_num_encoded'] = data['p_num_encoded'].fillna(-1)  # For unseen participants in test set
    return data

train_imputed['p_num'] = train['p_num']
test_imputed['p_num'] = test['p_num']
train_imputed = encode_participant(train_imputed, train)
test_imputed = encode_participant(test_imputed, train)

# Prepare final datasets
exclude_cols = {'id', 'time', target_col, 'p_num'}
features = [col for col in train_imputed.columns if col not in exclude_cols]

X = train_imputed[features]
y = train_imputed[target_col]
X_test = test_imputed[features]

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Cross-validation with GroupKFold
groups = train['p_num']
gkf = GroupKFold(n_splits=5)
print("data ready")
# Hyperparameter Optimization with Optuna for LightGBM
def objective(trial):
    params = {
        'objective': 'regression',
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
    for train_index, val_index in gkf.split(X_scaled, y, groups):
        X_train_fold, X_val_fold = X_scaled.iloc[train_index], X_scaled.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
        
        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[LightGBMPruningCallback(trial, 'rmse')],
        )
        
        y_pred = gbm.predict(X_val_fold, num_iteration=gbm.best_iteration)
        rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)

# Retrain model on full data with best hyperparameters
best_params = study.best_params
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['verbosity'] = -1

train_data = lgb.Dataset(X_scaled, label=y)

gbm = lgb.train(
    best_params,
    train_data,
    num_boost_round=gbm.best_iteration,
    valid_sets=[train_data],
    verbose_eval=100,
)

# LightGBM Predictions
lgbm_pred = gbm.predict(X_test_scaled)

# CatBoost Model
cat_features = [X.columns.get_loc(col) for col in X.columns if 'activity' in col or col == 'p_num_encoded']
cat_model = cb.CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=10,
    eval_metric='RMSE',
    random_seed=42,
    bagging_temperature=0.2,
    od_type='Iter',
    metric_period=50,
    od_wait=20
)

cat_model.fit(X, y, cat_features=cat_features, verbose=False)
cat_pred = cat_model.predict(X_test)

# Ensemble Predictions
final_pred = 0.6 * lgbm_pred + 0.4 * cat_pred

# Create submission file
submission = pd.DataFrame({'id': test['id'], 'bg+1:00': final_pred})
submission.to_csv('submission_best_model.csv', index=False)
