import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define RMSE values and load predictions
rmse_values = {
    'submission.csv': 2.8100,
    'submission_voting.csv': 2.7403,
    'submission_corrected.csv': 6.7908,
    'submission-withoutboruta.csv': 2.6878,
    'submission-lgbm.csv': 2.7071,
    'submission-lgbm-finetune.csv': 2.6547,
    'submission_ensemble_2.csv': 2.7724,
    'submission_lstm_2.csv': 4.1313,
    'submission_lstm.csv': 4.1752,
    'submission_weighted_avg.csv' : 2.6633,
    'submission_avg.csv': 2.6638 ,
    'submission_final_nn.csv' : 4.0364

}

# Load each model's predictions
files = list(rmse_values.keys())
submissions = [pd.read_csv(file) for file in files]

# Find common IDs across all files
common_ids = set(submissions[0]['id'])
for submission in submissions[1:]:
    common_ids &= set(submission['id'])
common_ids = sorted(common_ids)

# Align predictions by common IDs
aligned_submissions = []
for submission in submissions:
    aligned = submission[submission['id'].isin(common_ids)].sort_values(by='id').reset_index(drop=True)
    aligned_submissions.append(aligned)

# Prepare the DataFrame with predictions as features
predictions_df = pd.DataFrame({'id': aligned_submissions[0]['id']})
for i, aligned in enumerate(aligned_submissions):
    predictions_df[f'pred_{i}'] = aligned['bg+1:00']

# Prepare features (X) and target (y) based on the RMSE-weighted average
X = predictions_df.drop(columns=['id'])
y = np.average(X, axis=1, weights=[1 / rmse_values[file] for file in files])

# Split data for training meta-models
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_learners = [
    ('linear_regression', LinearRegression()),
    ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance')),
    ('lgbm', LGBMRegressor(n_estimators=500, learning_rate=0.05))
]

# Meta-model: Linear regression on top of base models
meta_model = LinearRegression()

# Stacking model
stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_model, cv=5, n_jobs=-1)
stacking_model.fit(X_train, y_train)

# Predictions and evaluation
y_val_pred = stacking_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse}")

# Use the full dataset to predict final outcomes
stacking_model.fit(X, y)
y_test_pred = stacking_model.predict(X)

# Save the final stacked predictions
submission = pd.DataFrame({
    'id': predictions_df['id'],
    'bg+1:00': y_test_pred
})
submission.to_csv('final_stacked_submission.csv', index=False)
