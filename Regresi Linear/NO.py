import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import timedelta

# Load data
df = pd.read_excel("dataseth.xlsx")
df.dropna(inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# Fitur dan target
features = ['temperature', 'humidity', 'distance', 'speed']
target = 'no'  # Target diganti menjadi NO

# Data latih (7 hari)
train_df = df[df['date'] <= pd.to_datetime("2025-05-11").date()]
X_train = train_df[features]
y_train = train_df[target]

# Latih model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi di data latih
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)

print("Evaluasi di Data Latih (7 Hari):")
print(f"R² (R-squared): {r2_train:.4f}")
print(f"MAE: {mae_train:.4f}")
print(f"MSE: {mse_train:.4f}")

# Simulasi prediksi 23 hari ke depan
start_date = pd.to_datetime("2025-05-12")
n_days = 23
future_data = pd.DataFrame()
source_dates = df['date'].unique()

for i in range(n_days):
    ref_date = source_dates[i % len(source_dates)]
    ref_data = df[df['date'] == ref_date]

    pattern_hours = sorted(ref_data['hour'].unique())
    ref_data = ref_data[ref_data['hour'].isin(pattern_hours)]

    new_date = start_date + timedelta(days=i)
    delta = new_date - pd.to_datetime(ref_date)

    new_data = ref_data.copy()
    new_data['timestamp'] += delta
    new_data['date'] = new_data['timestamp'].dt.date
    new_data['hour'] = new_data['timestamp'].dt.hour

    future_data = pd.concat([future_data, new_data])

# Prediksi NO untuk 23 hari ke depan
X_future = future_data[features]
y_future_pred = model.predict(X_future)
future_data['no_predicted'] = y_future_pred

# Evaluasi jika data aktual NO tersedia
if 'no' in future_data.columns:
    try:
        r2_future = r2_score(future_data['no'], future_data['no_predicted'])
        mae_future = mean_absolute_error(future_data['no'], future_data['no_predicted'])
        mse_future = mean_squared_error(future_data['no'], future_data['no_predicted'])

        print("\nEvaluasi di Data Prediksi (23 Hari):")
        print(f"R² (R-squared): {r2_future:.4f}")
        print(f"MAE: {mae_future:.4f}")
        print(f"MSE: {mse_future:.4f}")
    except:
        print("\nTidak ada data aktual untuk NO selama 23 hari ke depan — hanya prediksi ditampilkan.")
else:
    print("\nKolom NO aktual tidak tersedia untuk data 23 hari ke depan.")

# Total NO yang diprediksi
total_no_predicted = future_data['no_predicted'].sum()
print(f"\nTotal NO diprediksi selama 23 hari ke depan: {total_no_predicted:.2f} ppm")

# Visualisasi
plt.figure(figsize=(18, 5))
plt.plot(train_df['timestamp'], train_df['no'], label="NO Aktual (7 Hari)", color='blue')
plt.plot(future_data['timestamp'], future_data['no_predicted'], label="NO Prediksi (23 Hari)", color='red')
plt.xlabel("Waktu")
plt.ylabel("Kadar NO (ppm)")
plt.title("Prediksi Kadar NO Selama 23 Hari ke Depan + Data Aktual 7 Hari")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
