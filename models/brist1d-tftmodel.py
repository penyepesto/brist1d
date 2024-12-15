import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import Trainer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
# Read the data
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')
def preprocess_data(data):
    data = data.copy()
    # Convert 'time' to datetime
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')
    # Create a datetime index if not present
    data = data.sort_values(by=['p_num', 'time'])
    # Fill missing values
    bg_columns = [col for col in data.columns if col.startswith('bg-')]
    data[bg_columns] = data[bg_columns].interpolate(method='linear', limit_direction='both')
    # Fill 'hr', 'steps', 'cals'
    for prefix in ['hr', 'steps', 'cals']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        data[cols] = data[cols].interpolate(method='linear', limit_direction='both')
    # Fill 'insulin' and 'carbs' with 0
    for prefix in ['insulin', 'carbs']:
        cols = [col for col in data.columns if col.startswith(prefix)]
        data[cols] = data[cols].fillna(0)
    # Drop 'activity' columns
    activity_columns = [col for col in data.columns if col.startswith('activity')]
    data.drop(activity_columns, axis=1, inplace=True)
    return data
df = preprocess_data(df)
dt = preprocess_data(dt)
# Assign a unique time index
df['time_idx'] = df.groupby('p_num').cumcount()
dt['time_idx'] = dt.groupby('p_num').cumcount()

# Ensure 'p_num' is a string
df['p_num'] = df['p_num'].astype(str)
dt['p_num'] = dt['p_num'].astype(str)
def add_time_features(data):
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    return data

df = add_time_features(df)
dt = add_time_features(dt)
# Target variable
target = 'bg+1:00'

# Time-varying known real covariates (e.g., insulin, carbs)
time_varying_known_reals = ['time_idx', 'hour', 'day_of_week', 'month']

# Time-varying unknown real covariates (e.g., bg readings, hr, steps)
time_varying_unknown_reals = [col for col in df.columns if col not in ['id', 'p_num', 'time', 'time_idx', target] + time_varying_known_reals]

# Remove target from unknown reals if present
if target in time_varying_unknown_reals:
    time_varying_unknown_reals.remove(target)

# Static categorical variables
static_categoricals = ['p_num']
# Drop rows with NaNs in target
df = df.dropna(subset=[target])

# Define maximum prediction and encoder lengths
max_prediction_length = 1  # We are predicting 1 step ahead
max_encoder_length = 72  # Use past 6 hours (72 * 5 minutes = 360 minutes)

# Training cut-off
training_cutoff = df['time_idx'].max() - max_prediction_length

# Create training TimeSeriesDataSet
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx='time_idx',
    target=target,
    group_ids=['p_num'],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    static_categoricals=static_categoricals,
    target_normalizer=GroupNormalizer(groups=['p_num'])
)

# Create validation TimeSeriesDataSet
validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True
)

# Create dataloaders
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
# Create a TimeSeriesDataSet for the test data
test_dataset = TimeSeriesDataSet.from_dataset(training, dt, predict=True, stop_randomization=True)

# Create dataloader for the test data
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
# Define the TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,  # Quantile output
    loss=torch.nn.MSELoss(),
    reduce_on_plateau_patience=4
)
# Use PyTorch Lightning Trainer
max_epochs = 30  # Adjust based on your computational resources
early_stop_callback = torch.optim.lr_scheduler.ReduceLROnPlateau(tft.optimizer, patience=3, factor=0.5)

trainer = Trainer(
    max_epochs=max_epochs,
    gpus=0,  # Set to 1 if using GPU
    gradient_clip_val=0.1,
)

# Train the model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)
# Calculate validation RMSE
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = tft.predict(val_dataloader)
rmse = mean_squared_error(actuals.numpy(), predictions.numpy(), squared=False)
print(f'Validation RMSE: {rmse}')
# Predict on the test set
test_predictions = tft.predict(test_dataloader)
# Ensure predictions are aligned with test data
submission = dt[['id']].copy()
submission['bg+1:00'] = test_predictions.numpy().flatten()
submission.to_csv('submission-tft.csv', index=False)
