import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, precision_recall_curve
from lightgbm import LGBMClassifier, LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
MODEL_NAME = "ai_smart_grid_model6_finetuned"
RANDOM_STATE = 42
FIXED_THRESHOLD = 0.75

# --- Load dataset ---
df = pd.read_csv("smart_grid_dataset1.csv")
df.rename(columns={
    "Power Consumption (kW)": "Power_usage",
    "Overload Condition": "Overload",
    "Transformer Fault": "Fault_Indicator"
}, inplace=True)

if 'Current_Load' in df.columns and 'Capacity' in df.columns:
    df['Load_Ratio'] = df['Current_Load'] / (df['Capacity'] + 1e-6)

X = df.drop(columns=["Power_usage", "Overload", "Fault_Indicator", "Timestamp"])
y_power = df["Power_usage"]
y_overload = df["Overload"].astype(int)
y_fault = df["Fault_Indicator"].astype(int)

# --- Normalize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split data ---
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_scaled, y_power, test_size=0.2, random_state=RANDOM_STATE)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_scaled, y_overload, test_size=0.2, random_state=RANDOM_STATE)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled, y_fault, test_size=0.2, random_state=RANDOM_STATE)

# --- Train Fault Detection ---
print(f"Training Fault Detection Model with fixed threshold = {FIXED_THRESHOLD}...")
pos_weight = len(y_train_f[y_train_f == 0]) / len(y_train_f[y_train_f == 1])
model_fault = LGBMClassifier(
    class_weight=None,
    scale_pos_weight=pos_weight,
    num_leaves=20,
    max_depth=5,
    min_child_samples=100,
    min_split_gain=0.5,  # more conservative splits
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.8,
    n_estimators=300,
    random_state=RANDOM_STATE
)
model_fault.fit(X_train_f, y_train_f)
y_proba_f = model_fault.predict_proba(X_test_f)[:, 1]
y_pred_f = (y_proba_f > FIXED_THRESHOLD).astype(int)

# --- Fault Confusion Matrix ---
cm_f = confusion_matrix(y_test_f, y_pred_f)
tn, fp, fn, tp = cm_f.ravel()
print("\n📊 Fault Confusion Matrix:")
print(cm_f)
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
print(f"True Negatives:  {tn}")

# --- Train Overload Detection ---
print("Training Overload Detection Model...")
model_overload = LGBMClassifier(
    objective="binary",
    class_weight="balanced",
    learning_rate=0.05,
    max_depth=4,
    n_estimators=100,
    random_state=RANDOM_STATE
)
model_overload.fit(X_train_o, y_train_o)
y_pred_o = model_overload.predict(X_test_o)

# --- Power Usage Prediction ---
print("Training Power Usage Prediction Model...")
model_power = LGBMRegressor(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    random_state=RANDOM_STATE
)
model_power.fit(X_train_p, y_train_p)
y_pred_p = model_power.predict(X_test_p)

# --- Output Directory ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"{MODEL_NAME}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# --- Save Results ---
df_output = pd.DataFrame({
    "Actual_Power": y_test_p,
    "Predicted_Power": y_pred_p,
    "Actual_Overload": y_test_o,
    "Predicted_Overload": y_pred_o,
    "Actual_Fault": y_test_f,
    "Predicted_Fault": y_pred_f,
    "Fault_Probability": y_proba_f
})
df_output.to_excel(os.path.join(output_dir, "all_predictions.xlsx"), index=False)

# --- Confusion Matrix Plots ---
def plot_conf_matrix(y_true, y_pred, title, filename, labels):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

plot_conf_matrix(y_test_f, y_pred_f, "Fault Detection", "fault_confusion_matrix.png", ["Normal", "Fault"])
plot_conf_matrix(y_test_o, y_pred_o, "Overload Detection", "overload_confusion_matrix.png", ["Normal", "Overload"])

# --- PR Curve ---
precision, recall, _ = precision_recall_curve(y_test_f, y_proba_f)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.')
plt.axvline(FIXED_THRESHOLD, color='red', linestyle='--', label=f"Threshold = {FIXED_THRESHOLD}")
plt.title("Precision-Recall Curve (Fault Detection)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
plt.close()

# --- Probability Histogram ---
plt.figure(figsize=(6, 4))
plt.hist(y_proba_f, bins=50, color="skyblue")
plt.axvline(FIXED_THRESHOLD, color='red', linestyle='--', label=f"Threshold = {FIXED_THRESHOLD}")
plt.title("Fault Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fault_probability_histogram.png"), dpi=300)
plt.close()

# --- Save Performance Report ---
with open(os.path.join(output_dir, "performance_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Fixed Threshold: {FIXED_THRESHOLD:.4f}\n")
    f.write(f"Power Prediction MSE: {mean_squared_error(y_test_p, y_pred_p):.6f}\n\n")
    f.write("Fault Detection Report:\n")
    f.write(classification_report(y_test_f, y_pred_f, zero_division=0))
    f.write("\nOverload Detection Report:\n")
    f.write(classification_report(y_test_o, y_pred_o, zero_division=0))

print(f"\n✅ All results saved to: {output_dir}")
