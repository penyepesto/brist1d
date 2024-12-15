import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load the training data
train = pd.read_csv('train.csv')

# Load the test data
test = pd.read_csv('test.csv')

# Function to aggregate data into 15-minute intervals
def aggregate_15min(df):
    df_15min = pd.DataFrame()
    df_15min['id'] = df['id']
    df_15min['p_num'] = df['p_num']
    df_15min['time'] = df['time']

    # **Include 'bg+1:00' if it exists in the DataFrame**
    if 'bg+1:00' in df.columns:
        df_15min['bg+1:00'] = df['bg+1:00']

    # List of prefixes to aggregate
    prefixes = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals']

    for prefix in prefixes:
        cols = [col for col in df.columns if col.startswith(prefix) and '-' in col]
        # Aggregate every three columns (5 min intervals) into one (15 min interval)
        for i in range(0, len(cols), 3):
            cols_to_avg = cols[i:i+3]
            col_name = f"{prefix}_{i//3}_15min"
            df_15min[col_name] = df[cols_to_avg].mean(axis=1)
    
    # For 'activity', we'll take the mode (most frequent activity) in each 15 min interval
    activity_cols = [col for col in df.columns if col.startswith('activity')]
    for i in range(0, len(activity_cols), 3):
        cols_to_mode = activity_cols[i:i+3]
        col_name = f"activity_{i//3}_15min"
        df_15min[col_name] = df[cols_to_mode].mode(axis=1)[0]

    return df_15min

# Aggregate training data
train_15min = aggregate_15min(train)

# Aggregate test data
test_15min = aggregate_15min(test)

# Remove rows where more than half of the insulin values are zero
insulin_cols = [col for col in train_15min.columns if col.startswith('insulin')]
train_15min['insulin_zero_count'] = (train_15min[insulin_cols] == 0).sum(axis=1)
train_15min = train_15min[train_15min['insulin_zero_count'] < (len(insulin_cols) / 2)]
train_15min.drop('insulin_zero_count', axis=1, inplace=True)

# Similarly for test data
test_15min['insulin_zero_count'] = (test_15min[insulin_cols] == 0).sum(axis=1)
test_15min = test_15min[test_15min['insulin_zero_count'] < (len(insulin_cols) / 2)]
test_15min.drop('insulin_zero_count', axis=1, inplace=True)

# Feature Engineering: Create time-based features
def create_time_features(df):
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['minute'] = pd.to_datetime(df['time']).dt.minute
    df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

train_15min = create_time_features(train_15min)
test_15min = create_time_features(test_15min)

# Drop unnecessary columns
train_15min.drop(['id', 'p_num', 'time'], axis=1, inplace=True)
test_ids = test_15min['id']
test_15min.drop(['id', 'p_num', 'time'], axis=1, inplace=True)

# Handle categorical variables (activities)
activity_cols = [col for col in train_15min.columns if col.startswith('activity')]
train_activities = train_15min[activity_cols]
test_activities = test_15min[activity_cols]

# Convert categorical activity columns to numerical using One-Hot Encoding
train_15min = pd.get_dummies(train_15min, columns=activity_cols, dummy_na=True)
test_15min = pd.get_dummies(test_15min, columns=activity_cols, dummy_na=True)

# Align the train and test dataframes by columns
train_15min, test_15min = train_15min.align(test_15min, join='left', axis=1, fill_value=0)

# Prepare the target variable
y = train_15min['bg+1:00']

# Drop the target variable from training data
train_15min.drop('bg+1:00', axis=1, inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train_15min)
test_imputed = imputer.transform(test_15min)

# Standardize features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_imputed)
test_scaled = scaler.transform(test_imputed)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_scaled, y, test_size=0.2, random_state=42)

# Define the model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5]
}

gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)

# Train the model
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate on validation set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'Validation RMSE: {rmse}')

# Make predictions on the test set
test_predictions = best_model.predict(test_scaled)

# Prepare submission file
submission = pd.DataFrame({
    'id': test_ids,
    'bg+1:00': test_predictions
})

# Save submission file
submission.to_csv('submission-gbr.csv', index=False)

# Plot feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[-200:]  # Top 200 features
feature_names = train_15min.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
