import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dan siapkan data
df = pd.read_excel("dataseth.xlsx")
df.dropna(inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# Fitur dan target
features = ['temperature', 'humidity', 'distance', 'speed']
target = 'no'

# Data latih (7 hari)
train_df = df[df['date'] <= pd.to_datetime("2025-05-11").date()]
X = train_df[features].values
y = train_df[target].values.reshape(-1, 1)

# Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Buat dataset dengan sequence (24 jam sebelumnya untuk prediksi 1 jam ke depan)
sequence_length = 24
X_seq, y_seq = [], []

for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y_scaled[i+sequence_length])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Bangun model LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Latih model
model.fit(X_seq, y_seq, epochs=50, batch_size=16, verbose=1)

# Buat data untuk prediksi 23 hari ke depan
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

# Prediksi NO pada data future
X_future = future_data[features].values
X_future_scaled = scaler_X.transform(X_future)

# Bentuk input sequence untuk prediksi
predicted_no = []
window = X_scaled[-sequence_length:]  # ambil window terakhir dari data latih

for i in range(len(X_future_scaled)):
    window = np.append(window, [X_future_scaled[i]], axis=0)
    input_seq = window[-sequence_length:].reshape(1, sequence_length, X_scaled.shape[1])
    y_pred_scaled = model.predict(input_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    predicted_no.append(y_pred[0][0])

# Simpan hasil
future_data['no_predicted'] = predicted_no

# Total NO
total_no_predicted = future_data['no_predicted'].sum()
print(f"\nTotal NO diprediksi selama 23 hari ke depan (LSTM): {total_no_predicted:.2f} ppm")

# Plot hasil
plt.figure(figsize=(18, 5))
plt.plot(train_df['timestamp'], train_df['no'], label="NO Aktual (7 Hari)", color='blue')
plt.plot(future_data['timestamp'], future_data['no_predicted'], label="NO Prediksi LSTM (23 Hari)", color='green')
plt.xlabel("Waktu")
plt.ylabel("Kadar NO (ppm)")
plt.title("Prediksi LSTM Kadar NO Selama 23 Hari ke Depan + Data Aktual 7 Hari")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Prediksi pada data pelatihan
y_train_pred_scaled = model.predict(X_seq)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_train_actual = scaler_y.inverse_transform(y_seq)

# Hitung metrik
r2 = r2_score(y_train_actual, y_train_pred)
mse = mean_squared_error(y_train_actual, y_train_pred)
mae = mean_absolute_error(y_train_actual, y_train_pred)

print(f"\nEvaluasi pada data pelatihan:")
print(f"R² Score: {r2:.4f}")
print(f"MSE     : {mse:.4f}")
print(f"MAE     : {mae:.4f}")
