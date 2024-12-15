import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import optuna
from optuna.integration import TFKerasPruningCallback
import re
import warnings
warnings.filterwarnings('ignore')
# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Convert 'time' to datetime and extract time-based features
def extract_time_features(data):
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    # Cyclical transformation
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    return data

train_df = extract_time_features(train_df)
test_df = extract_time_features(test_df)

# Clean column names
def clean_column_names(data):
    data.columns = data.columns.str.replace(':', '_', regex=False)
    data.columns = data.columns.str.replace('+', 'plus', regex=False)
    data.columns = data.columns.str.replace('-', 'minus', regex=False)
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return data

train_df = clean_column_names(train_df)
test_df = clean_column_names(test_df)

# Update target column name
target_col = 'bgplus1_00'

# Map activities to numeric impact
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
    # Sum activity impact over the last hour
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

train_df = process_activity_data(train_df)
test_df = process_activity_data(test_df)
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

train_df = aggregate_to_15_min_intervals(train_df)
test_df = aggregate_to_15_min_intervals(test_df)
def remove_insulin_zero_rows(data):
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    # Count number of zero insulin values per row
    zero_counts = (data[insulin_cols] == 0).sum(axis=1)
    total_counts = data[insulin_cols].notnull().sum(axis=1)
    # Remove rows where more than half of the insulin values are zero
    data = data.loc[zero_counts <= (total_counts / 2)]
    return data.reset_index(drop=True)

train_df = remove_insulin_zero_rows(train_df)
# Interpolate missing values
def interpolate_missing_values(data):
    data.sort_values(['p_num', 'time'], inplace=True)
    data = data.groupby('p_num').apply(lambda group: group.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill'))
    return data.reset_index(drop=True)

train_df = interpolate_missing_values(train_df)
test_df = interpolate_missing_values(test_df)
# Create additional features based on relationships
def create_additional_features(data):
    # Insulin sensitivity: insulin / bg (avoid division by zero)
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    for ins_col, bg_col in zip(sorted(insulin_cols), sorted(bg_cols)):
        data[f'insulin_sensitivity_{ins_col.split("_")[1]}'] = data[ins_col] / (data[bg_col] + 1e-5)
    # Difference in bg over time
    bg_cols_sorted = sorted(bg_cols, key=lambda x: int(x.split('_')[1]))
    for i in range(len(bg_cols_sorted) - 1):
        data[f'bg_diff_{i}'] = data[bg_cols_sorted[i]] - data[bg_cols_sorted[i + 1]]
    # Rolling statistics
    data['bg_mean'] = data[bg_cols].mean(axis=1)
    data['bg_std'] = data[bg_cols].std(axis=1)
    data['insulin_sum'] = data[insulin_cols].sum(axis=1)
    data['carbs_sum'] = data[[col for col in data.columns if col.startswith('carbs_')]].sum(axis=1)
    return data

train_df = create_additional_features(train_df)
test_df = create_additional_features(test_df)
# Select features
exclude_cols = {'id', 'time', target_col, 'p_num'}
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# Impute remaining missing values
train_df[feature_cols] = train_df[feature_cols].fillna(0)
test_df[feature_cols] = test_df[feature_cols].fillna(0)

# Normalize the features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# Prepare sequences for LSTM
sequence_length = 6  # You can adjust this based on the data

def create_sequences(data, seq_length, feature_cols, target_col=None):
    sequences = []
    targets = []
    groups = []
    ids = []
    data_array = data[feature_cols].values
    if target_col:
        target_array = data[target_col].values
    p_nums = data['p_num'].values
    data_ids = data['id'].values
    for i in range(len(data) - seq_length + 1):
        # Check if the group (participant) changes in the sequence
        if p_nums[i] != p_nums[i + seq_length - 1]:
            continue
        seq = data_array[i:i+seq_length]
        sequences.append(seq)
        if target_col:
            targets.append(target_array[i + seq_length - 1])
        groups.append(p_nums[i + seq_length - 1])
        ids.append(data_ids[i + seq_length - 1])
    if target_col:
        return np.array(sequences), np.array(targets), np.array(groups), np.array(ids)
    else:
        return np.array(sequences), np.array(groups), np.array(ids)

# For training data
X_train, y_train, groups, _ = create_sequences(train_df, sequence_length, feature_cols, target_col)

# For test data
X_test, test_groups, test_ids = create_sequences(test_df, sequence_length, feature_cols)

# Normalize the features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
nsamples, nx, ny = X_train.shape
X_train_reshaped = X_train.reshape((nsamples * nx, ny))
X_train_reshaped = scaler.fit_transform(X_train_reshaped)
X_train = X_train_reshaped.reshape((nsamples, nx, ny))

nsamples_test, nx_test, ny_test = X_test.shape
X_test_reshaped = X_test.reshape((nsamples_test * nx_test, ny_test))
X_test_reshaped = scaler.transform(X_test_reshaped)
X_test = X_test_reshaped.reshape((nsamples_test, nx_test, ny_test))
""" 
# Hyperparameter Tuning with Optuna
import optuna
from optuna.integration import TFKerasPruningCallback

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    epochs = trial.suggest_int('epochs', 10, 50)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    l1_reg = trial.suggest_loguniform('l1_reg', 1e-6, 1e-2)
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-6, 1e-2)
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])

    # Build the model with the suggested hyperparameters
    def build_model(input_shape):
        model = models.Sequential()
        model.add(layers.LSTM(
            lstm_units,
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.LSTM(
            lstm_units // 2,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(
            lstm_units // 4,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        ))
        model.add(layers.Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Cross-validation
    gkf = GroupKFold(n_splits=3)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_model(input_shape)

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Add pruning callback
        pruning_callback = TFKerasPruningCallback(trial, monitor='val_loss')

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, pruning_callback],
            verbose=0
        )

        y_pred = model.predict(X_val).reshape(-1)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse_scores.append(rmse)

        # Report intermediate objective value
        trial.report(rmse, fold)

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(rmse_scores) """

""" # Run the optimization study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# Retrieve the best hyperparameters
best_params = study.best_trial.params """

# Build and train the final model with best hyperparameters
best_params = {'learning_rate': 0.0003, 
               'batch_size': 64, 
               'epochs': 30, 
               'dropout_rate': 0.15, 
               'l1_reg': 0.0005, 
               'l2_reg': 0.0001, 
               'lstm_units': 32}

def build_final_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(
        best_params['lstm_units'],
        input_shape=input_shape,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['l1_reg'], l2=best_params['l2_reg'])
    ))
    model.add(layers.Dropout(best_params['dropout_rate']))
    model.add(layers.LSTM(
        best_params['lstm_units'] // 2,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['l1_reg'], l2=best_params['l2_reg'])
    ))
    model.add(layers.Dropout(best_params['dropout_rate']))
    model.add(layers.Dense(
        best_params['lstm_units'] // 4,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=best_params['l1_reg'], l2=best_params['l2_reg'])
    ))
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
model = build_final_model(input_shape)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions and create submission
predictions = model.predict(X_test).reshape(-1)
submission = pd.DataFrame({'id': test_ids, 'bg+1:00': predictions})
submission.to_csv('submission_lstm_optimized_byme.csv', index=False)
