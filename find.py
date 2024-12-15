import pandas as pd

# CSV dosyalarını yükle
df1 = pd.read_csv("submission_LSTM_LAST.csv")
df2 = pd.read_csv("outputs/submission_with_aggregated_features.csv")

# Sadece birinci dosyada olup ikinci dosyada olmayan ID'leri bul
unique_ids_in_df1 = df1[~df1['id'].isin(df2['id'])]

# Sadece ikinci dosyada olup birinci dosyada olmayan ID'leri bul
unique_ids_in_df2 = df2[~df2['id'].isin(df1['id'])]

# Sonuçları görüntüle
print("Sadece birinci dosyada olan ID'ler:")
print(unique_ids_in_df1)

print("\nSadece ikinci dosyada olan ID'ler:")
print(unique_ids_in_df2)

