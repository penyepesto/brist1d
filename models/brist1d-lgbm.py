
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Read the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Convert 'time' to datetime
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
dt['time'] = pd.to_datetime(dt['time'], format='%H:%M:%S')

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
    lag_steps = [1, 2, 3, 4]  # Lag steps (5-minute intervals)

    for col in bg_columns:
        # Lagged features
        for lag in lag_steps:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    # Interaction terms
    data['bg_hr_interaction'] = data['bg-0:00'] * data['hr-0:00']
    data['insulin_carbs_interaction'] = data['insulin-0:00'] * data['carbs-0:00']
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
    all_columns.remove('p_num')

# Fill NaNs in X with median values
X = X.fillna(X.median())
X_test = X_test.fillna(X.median())

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Feature Selection using LightGBM's feature importance
temp_lgbm = LGBMRegressor(n_estimators=100, random_state=42)
temp_lgbm.fit(X_scaled, y)
feature_importances = temp_lgbm.feature_importances_

# Ensure that the length of feature_importances matches the number of features
print(f"Number of features in X: {X.shape[1]}")
print(f"Length of feature_importances: {len(feature_importances)}")

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
})

# Select top 50 features
top_features = importance_df.sort_values(by='importance', ascending=False).head(50)['feature'].tolist()

# Update X and X_test with selected features
X_selected = X[top_features]
X_test_selected = X_test[top_features]

# Scale again
scaler = StandardScaler()
X_selected_scaled = scaler.fit_transform(X_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42, shuffle=False)


# Initialize LightGBM model with early stopping
lgbm_model = LGBMRegressor(
    early_stopping_rounds = 1000,
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42
)

# Fit model with early stopping
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],

)

# Make predictions on validation set
y_pred_val = lgbm_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse}')

# Make predictions on test set
predictions = lgbm_model.predict(X_test_selected_scaled)

# Create submission
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission-lgbm-2.csv', index=False)
