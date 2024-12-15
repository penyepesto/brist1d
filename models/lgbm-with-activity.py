import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, train_test_split
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
def create_lag_features(data, lags, feature_cols):
    for col in feature_cols:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

lags = range(1, 13)
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = create_lag_features(df, lags, feature_cols)
dt = create_lag_features(dt, lags, feature_cols)

# Additional feature interactions
df['bg_insulin_interaction'] = df['bgminus0_00'] * df['insulinminus0_00']
dt['bg_insulin_interaction'] = dt['bgminus0_00'] * dt['insulinminus0_00']

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

# Feature Selection using LightGBM
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X, y)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 300  # Adjust N as needed
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

X = X[selected_features]
X_test = X_test[selected_features]

# Split data into train and validation sets based on participants
participants = df['p_num'].unique()
train_participants, val_participants = train_test_split(participants, test_size=0.2, random_state=42)
train_df = df[df['p_num'].isin(train_participants)]
val_df = df[df['p_num'].isin(val_participants)]

X_train = train_df[selected_features]
y_train = train_df[target_col]
X_val = val_df[selected_features]
y_val = val_df[target_col]

# Define best_params directly
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.005,
    'max_depth': 15,
    'n_estimators': 1000,
    'num_leaves': 31,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'subsample': 1.0,
    'random_state': 42,
    'verbose': -1,
}

# Train the model
lgbm_model = LGBMRegressor(**best_params)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['train', 'valid'],
    eval_metric='rmse',
    #verbose=20,
    #early_stopping_rounds=None  # No early stopping
)

# Plot training and validation loss
train_rmse = lgbm_model.evals_result_['train']['rmse']
valid_rmse = lgbm_model.evals_result_['valid']['rmse']

# Evaluate model performance
y_train_pred = lgbm_model.predict(X_train)
y_val_pred = lgbm_model.predict(X_val)

train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

print(f'Training RMSE: {train_rmse}')
print(f'Validation RMSE: {val_rmse}')

# Make predictions on the test set
predictions = lgbm_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_with_activity_4.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(train_rmse, label='Training RMSE')
plt.plot(valid_rmse, label='Validation RMSE')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training and Validation RMSE over Iterations')
plt.legend()
plt.show()