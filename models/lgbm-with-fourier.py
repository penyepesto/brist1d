import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import optuna
import matplotlib.pyplot as plt

# Read the data
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

# Apply Fourier transforms to capture seasonality
def create_fourier_features(data, period, order):
    t = data['time'].dt.hour + data['time'].dt.minute / 60
    for i in range(1, order + 1):
        data[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        data[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)
    return data

# Assume a daily period (24 hours) and use 3 harmonics
df = create_fourier_features(df, period=24, order=3)
dt = create_fourier_features(dt, period=24, order=3)

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
# Create lag features up to 12 lags (6 hours)
def create_lag_features(data, lags, feature_cols):
    for col in feature_cols:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

lags = range(1, 13)
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = create_lag_features(df, lags, feature_cols)
dt = create_lag_features(dt, lags, feature_cols)

# Create interaction terms
def create_interaction_terms(data, feature_pairs, lags):
    for (f1, f2) in feature_pairs:
        for lag in lags:
            col1 = f'{f1}_lag_{lag}' if lag > 0 else f'{f1}minus0_00'
            col2 = f'{f2}_lag_{lag}' if lag > 0 else f'{f2}minus0_00'
            interaction_col = f'{col1}_x_{col2}'
            if col1 in data.columns and col2 in data.columns:
                data[interaction_col] = data[col1] * data[col2]
    return data

# Define feature pairs for interaction
feature_pairs = [
    ('insulinminus0_00', 'carbsminus0_00'),
    ('insulinminus0_00', 'bgminus0_00'),
    ('carbsminus0_00', 'bgminus0_00')
]

# Create interaction terms
df = create_interaction_terms(df, feature_pairs, lags=range(0, 13))  # lags from 0 to 12
dt = create_interaction_terms(dt, feature_pairs, lags=range(0, 13))

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

# Ensure consistent column order
X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Impute remaining missing values
X = X.fillna(0)
X_test = X_test.fillna(0)

# Feature Selection using mutual information
selector = SelectKBest(mutual_info_regression, k=200)  
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)
X_test = X_test[selected_features]

# Scale the data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)
X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

# Define groups for cross-validation
groups = df['p_num']

# Hyperparameter Optimization using Optuna
import optuna
from sklearn.model_selection import GroupKFold
""" 
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.1, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'n_estimators': 5000,
        'random_state': 42,
        'verbose': -1
    }
    
    model = LGBMRegressor(**params)
    
    # Use GroupKFold for cross-validation
    gkf = GroupKFold(n_splits=5)
    rmse_scores = []
    
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            #early_stopping_rounds=50,
            #verbose=False
        )
        
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores) """

# Optimize hyperparameters
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=5)

#print('Best RMSE:', study.best_value)
#print('Best hyperparameters:', study.best_params)

# Train final model with best hyperparameters
#best_params = study.best_params

best_params =  {'num_leaves': 27, 
                'learning_rate': 0.003,#0.002383044524672185, 
                'max_depth': 30, 
                'reg_alpha': 0.948469941568012, 
                'reg_lambda': 1.093710806042597, 
                'subsample': 0.8704358973910173, 
                'colsample_bytree': 0.7437442540043924}
best_params['n_estimators'] = 2000  # Ensure enough estimators for early stopping
best_params['random_state'] = 42
best_params['verbose'] = 50
best_params['early_stopping_rounds']=50
final_model = LGBMRegressor(**best_params)
 
# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=10)

rmse_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        #early_stopping_rounds=50,
        #verbose=False
    )

    y_pred = final_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    print(f'Fold {fold+1} RMSE: {rmse}')

print(f'Average RMSE from cross-validation: {np.mean(rmse_scores)}')
 
# Train final model on full data
final_model.fit(
    X, y,
    eval_set=[(X, y)],
)

# Make predictions on the test set
predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('lgbm-with-fourier-2.csv', index=False)

print("Model training and prediction completed successfully!")