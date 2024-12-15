import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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
    # Cyclical transformation
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
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

# Step 4: Handle missing values and process activity data
# Instead of dropping 'activity' columns, we will process them

# Define activity impact mapping based on domain knowledge
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

# Function to process activity data
def process_activity_data(data):
    # Identify activity columns
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    
    # For each activity column, map the activity to its impact
    for col in activity_columns:
        # Map activities to their impact
        data[col] = data[col].map(activity_impact)
    
    # Replace NaN values in activity columns with 0 (no activity)
    data[activity_columns] = data[activity_columns].fillna(0)
    
    # Optionally, create aggregated activity features
    # For example, sum of activity impacts over the past 1 hour
    activity_cols_last_hour = [col for col in activity_columns if int(col.split('_')[1]) <= 12]  # Assuming _X_XX format
    data['activity_impact_sum'] = data[activity_cols_last_hour].sum(axis=1)
    
    # Drop the individual activity columns if not needed
    # data.drop(columns=activity_columns, inplace=True)
    
    return data

df = process_activity_data(df)
dt = process_activity_data(dt)

# Step 5: Fill missing values
def fill_missing_values(data):
    # Fill 'bg' columns
    bg_columns = [col for col in data.columns if col.startswith('bgminus') or col.startswith('bgplus')]
    data[bg_columns] = data[bg_columns].fillna(method='ffill').fillna(method='bfill')
    
    # Fill 'hr', 'steps', 'cals' with forward/backward fill
    for prefix in ['hr', 'steps', 'cals']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        data[cols] = data[cols].fillna(method='ffill').fillna(method='bfill')
    
    # Fill 'insulin' and 'carbs' with 0
    for prefix in ['insulin', 'carbs']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        data[cols] = data[cols].fillna(0)
    
    return data

df = fill_missing_values(df)
dt = fill_missing_values(dt)


# Step 6: Feature Engineering
# Interaction between activity impact and hour
df['activity_hour_interaction'] = df['activity_impact_sum'] * df['hour']
dt['activity_hour_interaction'] = dt['activity_impact_sum'] * dt['hour']

def create_activity_lag_features(data, lags):
    for lag in lags:
        data[f'activity_impact_lag_{lag}'] = data['activity_impact_sum'].shift(lag)
    return data

lags = range(1, 13)  # Adjust lags as needed
df = create_activity_lag_features(df, lags)
dt = create_activity_lag_features(dt, lags)

def create_activity_rolling_features(data, windows):
    for window in windows:
        data[f'activity_impact_rolling_mean_{window}'] = data['activity_impact_sum'].rolling(window, min_periods=1).mean()
        data[f'activity_impact_rolling_std_{window}'] = data['activity_impact_sum'].rolling(window, min_periods=1).std()
    return data

windows = [12, 24, 36]
df = create_activity_rolling_features(df, windows)
dt = create_activity_rolling_features(dt, windows)

def create_lag_features(data, lags, feature_cols):
    for col in feature_cols:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

# Extend lags to cover up to 3 hours (assuming 5-minute intervals)
lags = range(1, 37)
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00', 'hrminus0_00', 'stepsminus0_00', 'calsminus0_00']
df = create_lag_features(df, lags, feature_cols)
dt = create_lag_features(dt, lags, feature_cols)

def create_rolling_features(data, windows, feature_cols):
    for col in feature_cols:
        for window in windows:
            data[f'{col}_rolling_mean_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            data[f'{col}_rolling_std_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
            data[f'{col}_rolling_min_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
            data[f'{col}_rolling_max_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
            data[f'{col}_rolling_median_{window}'] = data.groupby('p_num')[col].rolling(window, min_periods=1).median().reset_index(level=0, drop=True)
    return data

windows = [12, 24, 36, 48, 60, 72]  # Rolling windows up to 6 hours
feature_cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00', 'hrminus0_00', 'stepsminus0_00', 'calsminus0_00']
df = create_rolling_features(df, windows, feature_cols)
dt = create_rolling_features(dt, windows, feature_cols)

def time_since_last_event(data, event_cols):
    for col in event_cols:
        event_occurred = data[col] > 0
        data[f'time_since_last_{col}'] = event_occurred.groupby(data['p_num']).cumsum().replace(0, np.nan)
        data[f'time_since_last_{col}'] = data.groupby('p_num')[f'time_since_last_{col}'].fillna(method='ffill').fillna(0)
    return data

event_cols = ['insulinminus0_00', 'carbsminus0_00']
df = time_since_last_event(df, event_cols)
dt = time_since_last_event(dt, event_cols)

def extract_additional_time_features(data):
    data['day_of_week'] = data['time'].dt.dayofweek
    data['day_of_month'] = data['time'].dt.day
    data['week_of_year'] = data['time'].dt.isocalendar().week
    # Cyclical encoding
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_month_sin'] = np.sin(2 * np.pi * data['day_of_month'] / 31)
    data['day_of_month_cos'] = np.cos(2 * np.pi * data['day_of_month'] / 31)
    data['week_of_year_sin'] = np.sin(2 * np.pi * data['week_of_year'] / 52)
    data['week_of_year_cos'] = np.cos(2 * np.pi * data['week_of_year'] / 52)
    return data

df = extract_additional_time_features(df)
dt = extract_additional_time_features(dt)


# Interactions between insulin and carbs
df['insulin_carbs_interaction'] = df['insulinminus0_00'] * df['carbsminus0_00']
dt['insulin_carbs_interaction'] = dt['insulinminus0_00'] * dt['carbsminus0_00']

def participant_stats(data, cols):
    participant_means = data.groupby('p_num')[cols].transform('mean')
    participant_stds = data.groupby('p_num')[cols].transform('std')
    data[[f'{col}_participant_mean' for col in cols]] = participant_means
    data[[f'{col}_participant_std' for col in cols]] = participant_stds
    return data

cols = ['bgminus0_00', 'insulinminus0_00', 'carbsminus0_00']
df = participant_stats(df, cols)
dt = participant_stats(dt, cols)

def interpolate_missing_values(data):
    data.sort_values(['p_num', 'time'], inplace=True)
    data = data.groupby('p_num').apply(lambda group: group.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill'))
    return data.reset_index(drop=True)

df = interpolate_missing_values(df)
dt = interpolate_missing_values(dt)


# Identify HIIT sessions
def tag_hiit_sessions(data):
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    for col in activity_columns:
        data[f'hiit_{col}'] = data[col].apply(lambda x: 1 if x == 'HIIT' else 0)
    return data

df = tag_hiit_sessions(df)
dt = tag_hiit_sessions(dt)

# Sum HIIT sessions over time windows
def sum_hiit_sessions(data, windows):
    hiit_cols = [col for col in data.columns if col.startswith('hiit_activity')]
    for window in windows:
        data[f'hiit_sum_{window}'] = data[hiit_cols].sum(axis=1)
    return data

windows = [12, 24, 36]
df = sum_hiit_sessions(df, windows)
dt = sum_hiit_sessions(dt, windows)

def categorize_bg_level(data):
    data['bg_category'] = pd.cut(data['bgminus0_00'], bins=[0, 4, 7.8, np.inf], labels=['low', 'normal', 'high'])
    data = pd.get_dummies(data, columns=['bg_category'])
    return data

df = categorize_bg_level(df)
dt = categorize_bg_level(dt)

# Simple insulin action model
df['expected_insulin_effect'] = df['insulinminus0_00'].rolling(window=12, min_periods=1).apply(lambda x: np.sum(x[::-1] * np.exp(-np.arange(len(x))/12)), raw=True)
dt['expected_insulin_effect'] = dt['insulinminus0_00'].rolling(window=12, min_periods=1).apply(lambda x: np.sum(x[::-1] * np.exp(-np.arange(len(x))/12)), raw=True)

# Step 7: Prepare data for modeling
# Exclude columns not needed
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - {'id', 'time', 'p_num'}

# Get the union of columns
all_columns = list(df_columns | dt_columns)

# Ensure both datasets have the same columns and order
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan
    if col not in dt.columns:
        dt[col] = np.nan
all_columns = sorted(all_columns)

X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# Step 8: Impute remaining missing values using KNNImputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Step 9: Scale the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns)

# Step 10: Feature Selection using LightGBM feature importances
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X_scaled, y)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 100  # Adjust N based on your preference
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

# Update X and X_test with selected features
X_selected = X_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# Step 11: Model Training with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

lgbm_model = LGBMRegressor(random_state=42, n_estimators=1000)

# Prepare for cross-validation
rmse_scores = []
for train_index, val_index in tscv.split(X_selected):
    X_train, X_val = X_selected.iloc[train_index], X_selected.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
#        early_stopping_rounds=50,
#        verbose=False
    )
    y_pred = lgbm_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)

print(f'Average RMSE from cross-validation: {np.mean(rmse_scores)}')

# Step 4: Hyperparameter Tuning with RandomizedSearchCV
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
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

lgbm_random.fit(X_selected, y)

print(f'Best RMSE: {-lgbm_random.best_score_}')
print(f'Best Parameters: {lgbm_random.best_params_}')

# Step 5: Train Final Model with Best Parameters
best_params = lgbm_random.best_params_
lgbm_final = LGBMRegressor(**best_params, random_state=42)
lgbm_final.fit(X_selected, y)

# Step 6: Make Predictions and Create Submission File
predictions = lgbm_final.predict(X_test_selected)

submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_with_activity.csv', index=False)

# Step 14: Analyze feature importances
feature_imp = pd.DataFrame({
    'Feature': selected_features,
    'Importance': lgbm_model.feature_importances_
})
feature_imp.sort_values('Importance', ascending=False, inplace=True)

plt.figure(figsize=(12,6))
plt.barh(feature_imp['Feature'][:20], feature_imp['Importance'][:20])
plt.gca().invert_yaxis()
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
