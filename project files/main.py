# Traffictelligence: Traffic Volume Prediction using ML

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/traffic_volume.csv')

# Preprocess
df = df.dropna()
X = df.drop(['traffic_volume'], axis=1)
y = df['traffic_volume']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.legend()
plt.title("Traffic Volume Prediction (Sample)")
plt.xlabel("Sample")
plt.ylabel("Volume")
plt.savefig("models/result_plot.png")
plt.show()
