import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

# Verileri yükleyin
df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

# Adım 1: Veri Ön İşleme
# 'time' sütununu datetime tipine çevirip zaman özelliklerini çıkaralım
def extract_time_features(data):
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    # Döngüsel dönüşüm
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 60)
    data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 60)
    return data

df = extract_time_features(df)
dt = extract_time_features(dt)

# Sütun isimlerini temizleyin
def clean_column_names(data):
    data.columns = data.columns.str.replace(':', '_', regex=False)
    data.columns = data.columns.str.replace('+', 'plus', regex=False)
    data.columns = data.columns.str.replace('-', 'minus', regex=False)
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    return data

df = clean_column_names(df)
dt = clean_column_names(dt)

# Hedef sütun adını güncelleyin
target_col = 'bgplus1_00'

# Aktivite verilerini işleyin
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

# Zaman offsetini dakikaya çeviren fonksiyon
def get_time_offset_minutes(col_name):
    match = re.search(r'minus(\d+)_(\d+)', col_name)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        total_minutes = hours * 60 + minutes
        return total_minutes
    else:
        return None

print("Adım 1 tamamlandı")

# Adım 2: Verileri 15 dakikalık aralıklara agregasyon yapın
def aggregate_to_15_min_intervals(data):
    feature_types = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
    for feature in feature_types:
        feature_cols = [col for col in data.columns if col.startswith(feature + 'minus')]
        # Her sütun için zaman offsetini alın
        col_time_offsets = {}
        for col in feature_cols:
            offset = get_time_offset_minutes(col)
            if offset is not None:
                col_time_offsets[col] = offset
        # Sütunları 15 dakikalık aralıklara gruplandırın
        bins = np.arange(0, 360 + 15, 15)  # 0'dan 360 dakikaya (6 saat), 15 adım
        bin_labels = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        bin_cols = {label: [] for label in bin_labels}
        for col, offset in col_time_offsets.items():
            for (start, end) in bin_labels:
                if start <= offset < end:
                    bin_cols[(start, end)].append(col)
                    break
        # Her aralık için ortalamayı hesaplayın
        for (start, end), cols in bin_cols.items():
            if len(cols) > 0:
                data[f'{feature}_{start}_{end}'] = data[cols].mean(axis=1)
            else:
                data[f'{feature}_{start}_{end}'] = np.nan
        # Orijinal sütunları düşürün
        data.drop(columns=feature_cols, inplace=True)
    return data

df = aggregate_to_15_min_intervals(df)
dt = aggregate_to_15_min_intervals(dt)

# Adım 3: İnsülin değerlerinin yarısından fazlası sıfır olan satırları kaldırın
def remove_insulin_zero_rows(data):
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    # Satır başına sıfır insülin değerlerinin sayısını sayın
    zero_counts = (data[insulin_cols] == 0).sum(axis=1)
    total_counts = data[insulin_cols].notnull().sum(axis=1)
    # İnsülin değerlerinin yarısından fazlası sıfır olan satırları kaldırın
    data = data.loc[zero_counts <= (total_counts / 2)]
    return data.reset_index(drop=True)

df = remove_insulin_zero_rows(df)
print("Adım 3 tamamlandı")

# Adım 4: Ek özellikler oluşturun
def create_additional_features(data):
    # Glukoz değerlerinin farkları
    bg_cols = [col for col in data.columns if col.startswith('bg_')]
    bg_cols_sorted = sorted(bg_cols, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for i in range(len(bg_cols_sorted)-1):
        data[f'bg_diff_{i}'] = data[bg_cols_sorted[i]] - data[bg_cols_sorted[i+1]]

    # Farklı pencere boyutlarında hareketli ortalamalar ve standart sapmalar
    window_sizes = [2, 4, 6, 8]  # Daha fazla pencere boyutu ekledik
    for window in window_sizes:
        data[f'bg_ma_{window}'] = data[bg_cols_sorted].rolling(window=window, axis=1).mean().iloc[:, -1]
        data[f'bg_std_{window}'] = data[bg_cols_sorted].rolling(window=window, axis=1).std().iloc[:, -1]

    # Üstel hareketli ortalamalar
    def exponential_moving_average(arr, alpha=0.3):
        ema = [arr[0]]
        for val in arr[1:]:
            ema.append(alpha * val + (1 - alpha) * ema[-1])
        return ema

    data['bg_ema'] = data[bg_cols_sorted].apply(lambda row: exponential_moving_average(row.values)[-1], axis=1)

    # Etkileşim terimleri
    insulin_cols = [col for col in data.columns if col.startswith('insulin_')]
    carbs_cols = [col for col in data.columns if col.startswith('carbs_')]
    data['insulin_sum'] = data[insulin_cols].sum(axis=1)
    data['carbs_sum'] = data[carbs_cols].sum(axis=1)
    data['insulin_carbs_ratio'] = data['insulin_sum'] / (data['carbs_sum'] + 1e-5)

    # Önceki glukoz seviyeleri
    data['bg_previous'] = data[bg_cols_sorted].iloc[:, -1]

    # Glukoz değişim hızı
    data['bg_rate_of_change'] = data[bg_cols_sorted].diff(axis=1).mean(axis=1)

    # Aktivite etkisi toplamı
    activity_cols = [col for col in data.columns if col.startswith('activity_')]
    data['activity_impact_sum'] = data[activity_cols].sum(axis=1)

    # Kalp hızı ve adım sayısı etkileşimi
    hr_cols = [col for col in data.columns if col.startswith('hr_')]
    steps_cols = [col for col in data.columns if col.startswith('steps_')]
    data['hr_mean'] = data[hr_cols].mean(axis=1)
    data['steps_sum'] = data[steps_cols].sum(axis=1)
    data['hr_steps_product'] = data['hr_mean'] * data['steps_sum']

    return data

df = create_additional_features(df)
dt = create_additional_features(dt)

# Modelleme için verileri hazırlayın
exclude_cols = {'id', 'time', target_col, 'p_num'}
df_columns = set(df.columns) - exclude_cols
dt_columns = set(dt.columns) - exclude_cols
all_columns = list(df_columns | dt_columns)

# Eğitim ve test verilerinde aynı sütunların olduğundan emin olun
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan
    if col not in dt.columns:
        dt[col] = np.nan

X = df[all_columns]
y = df[target_col]
X_test = dt[all_columns]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)

# Feature Selection based on importance
temp_lgbm = LGBMRegressor(n_estimators=1500, random_state=42)
temp_lgbm.fit(X, y)
feature_importances = temp_lgbm.feature_importances_

N = 200  # Özellik sayısını artırdık
indices = np.argsort(feature_importances)[-N:]
selected_features = [X.columns[i] for i in indices]

X = X[selected_features]
X_test = X_test[selected_features]

# Cross-validation with GroupKFold and early stopping
groups = df['p_num']
gkf = GroupKFold(n_splits=5)

best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.01,
    'max_depth': 15,
    'n_estimators': 3000,
    'num_leaves': 64,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': -1
}

lgbm_model = LGBMRegressor(**best_params)

rmse_scores = []
for fold, (train_index, val_index) in enumerate(gkf.split(X, y, groups)):
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

print(f'Ortalama RMSE: {np.mean(rmse_scores)}')

# Tüm veriyle modeli eğitin
lgbm_model.fit(
    X, y,
    eval_set=[(X, y)],
    callbacks=[early_stopping(stopping_rounds=50)]
)

# Test verisinde tahmin yapın
predictions = lgbm_model.predict(X_test)

# Tahminleri 0 ile 30 arasında sınırlayın
predictions = np.clip(predictions, 0, 30)

# Gönderim dosyasını oluşturun
submission = pd.DataFrame({'id': dt['id'], 'bg+1:00': predictions})
submission.to_csv('submission_improve-.csv', index=False)
print("Tamamlandı")