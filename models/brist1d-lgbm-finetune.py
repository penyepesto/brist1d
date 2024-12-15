# Step 1: Feature Engineering
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

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
    # Cyclical transformation
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    return data

df = extract_time_features(df)
dt = extract_time_features(dt)

# Define bg_columns for df (exclude 'bg+1:00') and dt separately
bg_columns_df = [col for col in df.columns if (col.startswith('bg-') or col.startswith('bg+')) and col != 'bg+1:00']
bg_columns_dt = [col for col in dt.columns if col.startswith('bg-') or col.startswith('bg+')]

# Modify the fill_missing_values function to accept bg_columns
def fill_missing_values(data, bg_columns):
    # Fill 'bg' columns
    if bg_columns:
        data[bg_columns] = data[bg_columns].fillna(method='ffill').fillna(method='bfill')
    # Fill 'hr', 'steps', 'cals'
    for prefix in ['hr', 'steps', 'cals']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        if cols:
            data[cols] = data[cols].fillna(method='ffill').fillna(method='bfill')
    # Fill 'insulin' and 'carbs' with 0
    for prefix in ['insulin', 'carbs']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        if cols:
            data[cols] = data[cols].fillna(0)
    # Drop 'activity' columns
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    data.drop(activity_columns, axis=1, inplace=True)
    return data

# Apply fill_missing_values to df and dt
df = fill_missing_values(df, bg_columns_df)
dt = fill_missing_values(dt, bg_columns_dt)

# Clean column names to remove special characters
def clean_column_names(df):
    df.columns = df.columns.str.replace(':', '_', regex=False)
    df.columns = df.columns.str.replace('+', 'plus', regex=False)
    df.columns = df.columns.str.replace('-', 'minus', regex=False)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return df

# Apply the cleaning function to both train and test data
df = clean_column_names(df)
dt = clean_column_names(dt)

# Update target column name after cleaning
target_col = 'bgplus1_00'  # Updated target name after cleaning

# Adjust exclude_cols to include 'p_num'
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

# Impute missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scale the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns)

# Feature Selection using LightGBM feature importances
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X_scaled, y)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 50  # Adjust N based on your preference
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

# Update X and X_test with selected features
X_selected = X_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# Step 3: Model Training with TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Initialize LightGBM with early stopping
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
submission.to_csv('submission-lgbm-finetune.csv', index=False)
