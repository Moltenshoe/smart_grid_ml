import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('smart_grid_dataset1.csv')

# Validate columns
required_columns = ['Transformer Fault', 'Overload Condition', 'Timestamp', 'Power Consumption (kW)']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

# Prepare features and targets for fault detection
X = df.drop(columns=['Transformer Fault', 'Overload Condition', 'Timestamp'])
y_fault = df['Transformer Fault']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Output folder
output_dir = f"results/isolation_forest_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Get actual fault ratio to guide IsolationForest
fault_ratio = (y_fault == 1).sum() / len(y_fault)
print(f"🔍 Detected fault ratio in dataset: {fault_ratio:.4f}")

# Train Isolation Forest (unsupervised anomaly detection)
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=fault_ratio,  # set based on actual fault rate
    random_state=42
)
iso_forest.fit(X_scaled)

# Predict anomalies (1 = normal, -1 = anomaly → we mark 1 for fault)
y_pred = iso_forest.predict(X_scaled)
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluate fault detection
print("Isolation Forest Report:")
print(classification_report(y_fault, y_pred))
plot_confusion_matrix(y_fault, y_pred, 'Isolation Forest Confusion Matrix', 'confusion_matrix_iso_forest.png')

# Save classification report to Excel
report_fault = classification_report(y_fault, y_pred, output_dict=True)
df_report_fault = pd.DataFrame(report_fault).transpose()
df_report_fault.to_excel(os.path.join(output_dir, 'fault_classification_report.xlsx'))

# ---------- Power Usage Prediction ----------
# Features and target for regression
X_reg = df.drop(columns=['Power Consumption (kW)', 'Transformer Fault', 'Overload Condition', 'Timestamp'])
y_reg = df['Power Consumption (kW)']

# Split and scale
from sklearn.model_selection import train_test_split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
scaler_reg = StandardScaler()
X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled = scaler_reg.transform(X_test_r)

# Train XGBRegressor
reg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
reg_model.fit(X_train_r_scaled, y_train_r)

# Predict power usage
y_pred_reg = reg_model.predict(X_test_r_scaled)

# Save actual vs predicted to Excel
df_results = pd.DataFrame({'Actual Power Usage': y_test_r.values, 'Predicted Power Usage': y_pred_reg})
df_results.to_excel(os.path.join(output_dir, 'power_prediction.xlsx'), index=False)

# Print regression metrics
mse = mean_squared_error(y_test_r, y_pred_reg)
r2 = r2_score(y_test_r, y_pred_reg)
print(f"\n🔧 Power Usage Prediction:\nMSE: {mse:.2f}, R²: {r2:.2f}")

print(f"\n✅ All results saved to: {output_dir}")
