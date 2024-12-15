import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.metrics import mean_squared_error
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import tensorflow as tf

# Helper function to create lagged, rolling, and interaction features
def create_features(df):
    # Lagged features
    for lag in range(1, 13):  # 1-hour lag with 5-minute intervals
        df[f'bg_lag_{lag}'] = df.groupby('p_num')['bg'].shift(lag)
    
    # Rolling features
    for window in [3, 6, 12]:
        df[f'bg_rolling_mean_{window}'] = df['bg'].rolling(window=window).mean()
        df[f'bg_rolling_std_{window}'] = df['bg'].rolling(window=window).std()
    
    # Interaction terms
    df['insulin_carbs_interaction'] = df['insulin'] * df['carbs']
    
    # Cyclical time-based features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

# Loading and preprocessing data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')
df = create_features(df)
dt = create_features(dt)

# Encoding categorical data
le = LabelEncoder()
df['p_num'] = le.fit_transform(df['p_num'])
dt['p_num'] = le.transform(dt['p_num'])

# Target transformation
scaler = StandardScaler()
power_transformer = PowerTransformer()  # For stable distribution
df['target'] = power_transformer.fit_transform(df[['target']])
y = df.pop('target')
X = df.drop(columns=['time', 'id'])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Function to build LSTM model with Batch Normalization and Dropout for regularization
def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(LSTM(units, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Preparing LSTM data
X_train_lstm = np.reshape(scaler.fit_transform(X_train), (X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = np.reshape(scaler.transform(X_val), (X_val.shape[0], 1, X_val.shape[1]))

# LSTM model training with early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

lstm_model = build_lstm_model(input_shape=(1, X_train.shape[1]))
lstm_model.fit(X_train_lstm, y_train, epochs=500, batch_size=64, 
               validation_data=(X_val_lstm, y_val), 
               callbacks=[early_stopping, reduce_lr])

# LSTM Hold-out evaluation
y_pred_lstm = power_transformer.inverse_transform(lstm_model.predict(X_val_lstm))
rmse_lstm = np.sqrt(mean_squared_error(y_val, y_pred_lstm))
print(f'Hold-Out RMSE (LSTM): {rmse_lstm}')

# Optuna hyperparameter tuning for LightGBM
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-4, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 10.0)
    }
    gbm = lgb.train(param, lgb.Dataset(X_train, label=y_train), valid_sets=[lgb.Dataset(X_val, label=y_val)], 
                    early_stopping_rounds=30, verbose_eval=False)
    preds = gbm.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params

# Train final LightGBM model
final_lgb = lgb.train(best_params, lgb.Dataset(X_train, label=y_train), 
                      valid_sets=[lgb.Dataset(X_val, label=y_val)],
                      early_stopping_rounds=30)

# Predictions with LightGBM
y_pred_lgb = power_transformer.inverse_transform(final_lgb.predict(X_val).reshape(-1, 1))
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
print(f'Hold-Out RMSE (LightGBM): {rmse_lgb}')

# Ensemble with meta-model
stacked_preds_train = np.column_stack([y_pred_lstm, y_pred_lgb])
meta_model = Ridge()
meta_model.fit(stacked_preds_train, y_val)

# Ensemble predictions
y_pred_meta = meta_model.predict(stacked_preds_train)
rmse_meta = np.sqrt(mean_squared_error(y_val, y_pred_meta))
print(f'Hold-Out RMSE (Meta Model): {rmse_meta}')

# Testing predictions
X_test = scaler.transform(dt.drop(columns=['time', 'id']))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Predictions
y_test_lstm = lstm_model.predict(X_test_lstm).flatten()
y_test_lgb = final_lgb.predict(X_test)
stacked_test_preds = np.column_stack([y_test_lstm, y_test_lgb])

# Meta-model predictions on test
y_test_meta = meta_model.predict(stacked_test_preds)

# Saving predictions
submission = pd.DataFrame({'id': dt['id'], 'target': y_test_meta})
submission.to_csv('submission_lstm_3.csv', index=False)
