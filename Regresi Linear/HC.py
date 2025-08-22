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
target = 'hc'  # Ubah target dari 'no' ke 'hc'

# Data latih (7 hari)
train_df = df[df['date'] <= pd.to_datetime("2025-05-11").date()]
X_train = train_df[features]
y_train = train_df[target]

# Latih model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi model di data latih
y_train_pred = model.predict(X_train)
r2 = r2_score(y_train, y_train_pred)
mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)

print(f"R² (R-squared): {r2:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")

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

# Prediksi HC
X_future = future_data[features]
y_future_pred = model.predict(X_future)
future_data['hc_predicted'] = y_future_pred

# Hitung total HC
total_hc_predicted = future_data['hc_predicted'].sum()
print(f"\nTotal HC diprediksi selama 23 hari ke depan: {total_hc_predicted:.2f} ppm")

# Plot grafik
plt.figure(figsize=(18, 5))
plt.plot(train_df['timestamp'], train_df['hc'], label="HC Aktual (7 Hari)", color='blue')
plt.plot(future_data['timestamp'], future_data['hc_predicted'], label="HC Prediksi (23 Hari)", color='red')
plt.xlabel("Waktu")
plt.ylabel("Kadar HC (ppm)")
plt.title("Prediksi Kadar HC Selama 23 Hari ke Depan + Data Aktual 7 Hari")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
