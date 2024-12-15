
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
from sklearn.model_selection import train_test_split

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
# Create lag features up to 72 lags (6 hours)
def create_lag_features(data, lags, feature_cols):
    for col in feature_cols:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('p_num')[col].shift(lag)
    return data

lags = range(1, 73)
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


# Define essential features
features = [
    'bgminus0_00', 'insulinminus0_00', 'carbsminus0_00',
    'hour_sin', 'hour_cos', 'activity_impact_sum'
]

# Ensure features exist in both df and dt
for col in features:
    if col not in df.columns:
        df[col] = 0
    if col not in dt.columns:
        dt[col] = 0

# Prepare data
X = df[features].fillna(0)
y = df[target_col]
X_test = dt[features].fillna(0)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X = pd.DataFrame(X_scaled, columns=features)
X_test = pd.DataFrame(X_test_scaled, columns=features)

# Create sequences
seq_length = 12
def create_sequences(X, seq_length):
    X_seq = []
    for i in range(seq_length, len(X)):
        X_seq.append(X.iloc[i - seq_length:i].values)
    return np.array(X_seq)

X_seq = create_sequences(X, seq_length)
y_seq = y.iloc[seq_length:].values
X_test_seq = create_sequences(X_test, seq_length)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.1, shuffle=False
)

input_shape = (X_train.shape[1], X_train.shape[2])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tcn import TCN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Build the model with reduced complexity and increased regularization
model = Sequential([
    Input(shape=input_shape),
    TCN(
        nb_filters=32,
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        dropout_rate=0.5,
        return_sequences=False  # Removed kernel_regularizer
    ),
    Dense(1, kernel_regularizer=l2(0.005))  # Apply regularization here
])

# Compile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.003), loss='mse', metrics=['mae'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,  # Reduced batch size if necessary
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Evaluate on validation set
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f'Validation Loss (MSE): {val_loss}')
print(f'Validation MAE: {val_mae}')

# Make predictions on test data
predictions = model.predict(X_test_seq)

# Adjust submission IDs to align with sequenced data
submission_ids = dt['id'].iloc[seq_length:].values  # Start after seq_length rows

# Create submission DataFrame
submission = pd.DataFrame({'id': submission_ids, 'bg+1:00': predictions.flatten()})

# Check and fill missing rows if necessary
if len(submission) < 3644:
    # Fill the missing rows at the start if allowed, else leave them as NaN
    missing_rows = 3644 - len(submission)
    padding = pd.DataFrame({'id': dt['id'].iloc[:missing_rows].values, 'bg+1:00': np.nan})
    submission = pd.concat([padding, submission], ignore_index=True)

submission['bg+1:00'] = submission['bg+1:00'].fillna(submission['bg+1:00'].mean())
# Save to CSV
submission.to_csv('submission_tcn.csv', index=False)
print("TCN model training and prediction completed successfully!")