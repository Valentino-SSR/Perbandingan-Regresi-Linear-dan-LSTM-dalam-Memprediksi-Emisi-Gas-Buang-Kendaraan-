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
target = 'co'

# Data latih (7 hari)
train_df = df[df['date'] <= pd.to_datetime("2025-05-11").date()]
X = train_df[features].values
y = train_df[target].values.reshape(-1, 1)

# Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Buat dataset dengan sequence (24 titik = 2 menit sebelumnya, interval 5 detik)
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

# Prediksi CO pada data future
X_future = future_data[features].values
X_future_scaled = scaler_X.transform(X_future)

# Bentuk input sequence untuk prediksi
predicted_co = []
window = X_scaled[-sequence_length:]  # ambil window terakhir dari data latih

for i in range(len(X_future_scaled)):
    window = np.append(window, [X_future_scaled[i]], axis=0)
    input_seq = window[-sequence_length:].reshape(1, sequence_length, X_scaled.shape[1])
    y_pred_scaled = model.predict(input_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    predicted_co.append(y_pred[0][0])

    # opsional: bisa update X_future_scaled[i] dengan prediksi, untuk prediksi berantai

# Simpan hasil
future_data['co_predicted'] = predicted_co

# Total CO
total_co_predicted = future_data['co_predicted'].sum()
print(f"\nTotal CO diprediksi selama 23 hari ke depan (LSTM): {total_co_predicted:.2f} ppm")

# Plot hasil
plt.figure(figsize=(18, 5))
plt.plot(train_df['timestamp'], train_df['co'], label="CO Aktual (7 Hari)", color='blue')
plt.plot(future_data['timestamp'], future_data['co_predicted'], label="CO Prediksi LSTM (23 Hari)", color='green')
plt.xlabel("Waktu")
plt.ylabel("Kadar CO (ppm)")
plt.title("Prediksi LSTM Kadar CO Selama 23 Hari ke Depan + Data Aktual 7 Hari")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
