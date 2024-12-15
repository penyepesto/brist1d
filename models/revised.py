import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
    data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 60)
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
    'HIIT': -1,
    'Aerobic Workout': -1,
    'Hike': -1,
    'Zumba': -1,
    'Yoga': -1,
    'Swimming': -1,
    'Weights': -1,
    'Tennis': -1,
    'Sport': -1,
    'Workout': -1,
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

print("Step 1 completed")

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
print("Step 3 completed")

def create_additional_features(data):
    # Sum of insulin and carbs over last 6 hours
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    carbs_cols = [col for col in data.columns if col.startswith('carbs_')]
    data['insulin_sum'] = data[insulin_cols].sum(axis=1)
    data['carbs_sum'] = data[carbs_cols].sum(axis=1)
    data['insulin_carbs_ratio'] = data['insulin_sum'] / (data['carbs_sum'] + 1e-5)

    # Interaction terms
    data['insulin_carbs_interaction'] = data['insulin_sum'] * data['carbs_sum']

    # Lag features for bg, insulin, and carbs
    lag_minutes = [15, 30, 45, 60, 75, 90, 105, 120]  # Lags at every 15-minute interval
    for lag in lag_minutes:
        bg_col = f'bg_{lag}_{lag+15}'
        insulin_col = f'insulin_{lag}_{lag+15}'
        carbs_col = f'carbs_{lag}_{lag+15}'
        data[f'bg_lag_{lag}'] = data[bg_col] if bg_col in data.columns else np.nan
        data[f'insulin_lag_{lag}'] = data[insulin_col] if insulin_col in data.columns else np.nan
        data[f'carbs_lag_{lag}'] = data[carbs_col] if carbs_col in data.columns else np.nan

    # Previous glucose levels
    bg_cols = [col for col in data.columns if col.startswith('bg_') and col not in [f'bg_lag_{lag}' for lag in lag_minutes]]
    if bg_cols:
        data['bg_previous'] = data[bg_cols].iloc[:, -1]
    else:
        data['bg_previous'] = np.nan

    # Rate of change of glucose
    if len(bg_cols) >= 2:
        data['bg_rate_of_change'] = data[bg_cols].diff(axis=1).mean(axis=1)
    else:
        data['bg_rate_of_change'] = 0

    # Activity impact sum
    activity_cols = [col for col in data.columns if col.startswith('activity_')]
    if activity_cols:
        data['activity_impact_sum'] = data[activity_cols].sum(axis=1)
    else:
        data['activity_impact_sum'] = 0

    return data


df = create_additional_features(df)
dt = create_additional_features(dt)

# Step 5: Handle missing data
# Use SimpleImputer to fill missing values with median
imputer = SimpleImputer(strategy='median')
df_imputed = df.copy()
dt_imputed = dt.copy()

cols_to_impute = df.columns.difference(['id', 'time', 'p_num', target_col])

df_imputed[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
dt_imputed[cols_to_impute] = imputer.transform(dt[cols_to_impute])

# Step 6: Handle outliers in the target variable
# Clip target variable to reasonable range (e.g., 2.5 to 25 mmol/L)
df_imputed[target_col] = df_imputed[target_col].clip(lower=2.5, upper=25)

# Step 7: Prepare data for modeling
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df_imputed.columns) - exclude_cols
dt_columns = set(dt_imputed.columns) - exclude_cols
all_columns = list(df_columns | dt_columns)

# Ensure columns in training and test data match
for col in all_columns:
    if col not in df_imputed.columns:
        df_imputed[col] = 0
    if col not in dt_imputed.columns:
        dt_imputed[col] = 0

X = df_imputed[all_columns]
y = df_imputed[target_col]
X_test = dt_imputed[all_columns]

# Feature Selection based on importance
temp_lgbm = LGBMRegressor(n_estimators=500, random_state=42)
temp_lgbm.fit(X, y)
feature_importances = temp_lgbm.feature_importances_

N = 200  # Select top 100 features
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

X = X[selected_features]
X_test = X_test[selected_features]

# Step 8: Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.005,
    'max_depth': 15,
    'n_estimators': 5000,
    'num_leaves': 31,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': -1
}

lgbm_model = LGBMRegressor(**best_params)

rmse_scores = []
for fold, (train_index, val_index) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50)]
    )

    y_pred = lgbm_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse_scores.append(rmse)
    print(f'Fold {fold+1} RMSE: {rmse}')

print(f'Average RMSE: {np.mean(rmse_scores)}')

# Train final model on full data
lgbm_model.fit(
    X, y,
    eval_set=[(X, y)],
    callbacks=[early_stopping(stopping_rounds=50)]
)

# Make predictions on the test set
predictions = lgbm_model.predict(X_test)

# Clip predictions to reasonable range
predictions = np.clip(predictions, 2.5, 25)

# Create submission file
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_revised.csv', index=False)
print("Completed")