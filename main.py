import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("5G_V2N_Communication_Dataset.csv")

print("Dataset Preview:")
print(data.head())

# =========================
# 2. Select Numeric Columns
# =========================
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Target (you can change index if needed)
y = numeric_data.iloc[:, -2]   # second last column
X = numeric_data.drop(numeric_data.columns[-2], axis=1)

# =========================
# 3. Normalize Data
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 5. Model (Improved)
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 6. Prediction
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. Evaluation
# =========================
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# =========================
# 8. FINAL GRAPH (BEST ONE)
# =========================
plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred)

# Ideal prediction line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Final Model)")

plt.savefig("graph.png")
plt.show()