from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. بارگذاری داده‌ها
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip"
data = pd.read_csv(data_url, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False, na_values=['?'], index_col='datetime')
data.fillna(method='ffill', inplace=True)

# 2. انتخاب ویژگی‌ها و متغیر هدف
power_data = data['Global_active_power'].astype('float32')
power_data = power_data.resample('D').mean()

# 3. آماده‌سازی داده‌ها
scaler = MinMaxScaler()
power_scaled = scaler.fit_transform(power_data.values.reshape(-1, 1))

X, y = [], []
for i in range(30, len(power_scaled)):
    X.append(power_scaled[i-30:i, 0])
    y.append(power_scaled[i, 0])
X, y = np.array(X), np.array(y)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ساخت مدل
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# 5. ارزیابی مدل
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# 6. مصورسازی نتایج
plt.figure(figsize=(14, 8))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='orange')
plt.title("Actual vs Predicted Power Consumption")
plt.xlabel("Time")
plt.ylabel("Power Consumption (scaled)")
plt.legend()
plt.show()
