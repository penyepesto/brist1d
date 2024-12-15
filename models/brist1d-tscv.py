import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from bayes_opt import BayesianOptimization

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

    # Interaction terms
    data['bg_hr_interaction'] = data['bg-0:00'] * data['hr-0:00']
    data['bg_steps_interaction'] = data['bg-0:00'] * data['steps-0:00']
    data['insulin_carbs_interaction'] = data['insulin-0:00'] * data['carbs-0:00']

    # Fill any remaining NaNs
    data = data.fillna(method='bfill').fillna(method='ffill')
    return data

# Apply feature engineering
df = feature_engineering(df)
dt = feature_engineering(dt)

# Clean column names to remove special characters
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return df

# Apply the cleaning function to both train and test data
df = clean_column_names(df)
dt = clean_column_names(dt)

# Extract features and target after cleaning
target_col = 'bg_1_00'  # Updated target name after removing special characters
X = df.drop(columns=['id', 'time', target_col])
y = df[target_col]
X_test = dt.drop(columns=['id', 'time'])

# Ensure both datasets have the same columns after cleaning
all_columns = sorted(list(set(X.columns) | set(X_test.columns)))
for col in all_columns:
    if col not in X.columns:
        X[col] = np.nan
    if col not in X_test.columns:
        X_test[col] = np.nan

# Reorder columns for consistency
X = X[all_columns]
X_test = X_test[all_columns]

# Drop 'p_num' if it's in the features, as the test data has unseen participants
if 'p_num' in X.columns:
    X = X.drop(columns=['p_num'])
    X_test = X_test.drop(columns=['p_num'])

# Impute missing values
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Feature Selection
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X_scaled, y)
feature_importances = temp_lgbm.feature_importances_

# Select top N features
N = 50  # Adjust as needed
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

# Update X and X_test with selected features
X_selected = pd.DataFrame(X_scaled, columns=X.columns)[selected_features]
X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[selected_features]

# Define and fit the ensemble model
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit

# Define models
lgbm_model = LGBMRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
cat_model = CatBoostRegressor(random_state=42, verbose=0)

# Fit individual models
lgbm_model.fit(X_selected, y)
xgb_model.fit(X_selected, y)
cat_model.fit(X_selected, y)

# Ensemble with stacking
estimators = [
    ('lgbm', lgbm_model),
    ('xgb', xgb_model),
    ('cat', cat_model)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LGBMRegressor(random_state=42),
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1
)
stacking_model.fit(X_selected, y)

# Make predictions on the test set
ensemble_pred = stacking_model.predict(X_test_selected)

# Create submission
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': ensemble_pred})
submission.to_csv('submission-tscv.csv', index=False)
print("Submission file created: submission-tscv.csv")