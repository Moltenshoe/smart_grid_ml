import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
df = pd.read_csv("smart_grid_dataset1.csv")

# --- Rename columns ---
df.rename(columns={
    "Power Consumption (kW)": "Power_usage",
    "Overload Condition": "Overload",
    "Transformer Fault": "Fault_Indicator"
}, inplace=True)

# --- Check for missing columns ---
required_cols = ["Power_usage", "Overload", "Fault_Indicator"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"⚠️ Missing columns in dataset: {missing_cols}")
    exit()

# --- Feature and target split ---
X = df.drop(columns=["Power_usage", "Overload", "Fault_Indicator", "Timestamp"])
y_power = df["Power_usage"]
y_overload = df["Overload"]
y_fault = df["Fault_Indicator"]

# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-test splits ---
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_scaled, y_power, test_size=0.2, random_state=42)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_scaled, y_overload, test_size=0.2, random_state=42)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled, y_fault, test_size=0.2, random_state=42)

# --- Train models ---
model_power = LinearRegression().fit(X_train_p, y_train_p)
model_overload = RandomForestClassifier().fit(X_train_o, y_train_o)
model_fault = RandomForestClassifier().fit(X_train_f, y_train_f)

# --- Predictions ---
y_pred_p = model_power.predict(X_test_p)
y_pred_o = model_overload.predict(X_test_o)
y_pred_f = model_fault.predict(X_test_f)

# --- Evaluation ---
mse = mean_squared_error(y_test_p, y_pred_p)
print(f"[Power Prediction] Mean Squared Error: {mse:.6f}")

# --- Output path (based on script name) ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("outputs", f"{script_name}_outputs_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# --- Save power prediction ---
df_output = pd.DataFrame({"Actual": y_test_p, "Predicted": y_pred_p})
df_output.to_excel(os.path.join(output_dir, "power_prediction.xlsx"), index=False)

plt.figure(figsize=(8, 4))
plt.plot(y_test_p.values[:100], label="Actual")
plt.plot(y_pred_p[:100], label="Predicted")
plt.title("Power Usage Prediction (First 100 Samples)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "power_prediction_plot.png"))
plt.close()

# --- Overload report ---
acc_o = model_overload.score(X_test_o, y_test_o)
print(f"[Overload] Accuracy: {acc_o:.4f}")
print(classification_report(y_test_o, y_pred_o, zero_division=0))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_o, y_pred_o), annot=True, fmt="d", cmap="Blues")
plt.title("Overload Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "overload_confusion_matrix.png"))
plt.close()

# --- Fault report ---
acc_f = model_fault.score(X_test_f, y_test_f)
print(f"[Fault] Accuracy: {acc_f:.4f}")
print(classification_report(y_test_f, y_pred_f, zero_division=0))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_f, y_pred_f), annot=True, fmt="d", cmap="Reds")
plt.title("Fault Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fault_confusion_matrix.png"))
plt.close()

print(f"\n✅ All results saved to folder: {output_dir}")
