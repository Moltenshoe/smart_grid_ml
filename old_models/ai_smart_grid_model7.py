import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, precision_recall_curve
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
MODEL_NAME = "ai_smart_grid_model7_adasyn"
RANDOM_STATE = 42
FALLBACK_THRESHOLD = 0.75
FP_LIMIT = 500
FN_LIMIT = 150

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_scaled, y_power, test_size=0.2, random_state=RANDOM_STATE)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_scaled, y_overload, test_size=0.2, random_state=RANDOM_STATE)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled, y_fault, test_size=0.2, random_state=RANDOM_STATE)

# --- Fault Detection with ADASYN ---
print("Training Fault Detection Model with ADASYN + LightGBM...")
adasyn = ADASYN(random_state=RANDOM_STATE)
X_train_f_res, y_train_f_res = adasyn.fit_resample(X_train_f, y_train_f)

model_fault = LGBMClassifier(
    class_weight='balanced',
    max_depth=5,
    learning_rate=0.05,
    n_estimators=100,
    random_state=RANDOM_STATE
)
model_fault.fit(X_train_f_res, y_train_f_res)
y_proba_f = model_fault.predict_proba(X_test_f)[:, 1]

# --- Smart Thresholding ---
thresholds = np.linspace(0.99, 0.01, 100)
best_threshold = FALLBACK_THRESHOLD
for t in thresholds:
    preds = (y_proba_f > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_f, preds).ravel()
    if fp <= FP_LIMIT and fn <= FN_LIMIT:
        best_threshold = t
        break
y_pred_f = (y_proba_f > best_threshold).astype(int)

# --- Overload Model ---
print("Training Overload Detection Model...")
model_overload = LGBMClassifier(
    objective='binary',
    class_weight='balanced',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    random_state=RANDOM_STATE
)
model_overload.fit(X_train_o, y_train_o)
y_pred_o = model_overload.predict(X_test_o)

# --- Power Prediction Model ---
print("Training Power Usage Prediction Model...")
model_power = LGBMRegressor(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    random_state=RANDOM_STATE
)
model_power.fit(X_train_p, y_train_p)
y_pred_p = model_power.predict(X_test_p)

# --- Output Folder ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"{MODEL_NAME}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# --- Save Predictions ---
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
plt.axvline(best_threshold, color='red', linestyle='--', label=f"Threshold = {best_threshold:.2f}")
plt.title("Precision-Recall Curve (Fault Detection)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
plt.close()

# --- Histogram ---
plt.figure(figsize=(6, 4))
plt.hist(y_proba_f, bins=50, color="skyblue")
plt.axvline(best_threshold, color='red', linestyle='--', label=f"Threshold = {best_threshold:.2f}")
plt.title("Fault Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fault_probability_histogram.png"), dpi=300)
plt.close()

# --- Performance Report ---
with open(os.path.join(output_dir, "performance_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Threshold Used: {best_threshold:.4f}\n")
    f.write(f"Power Prediction MSE: {mean_squared_error(y_test_p, y_pred_p):.6f}\n\n")
    f.write("Fault Detection Report:\n")
    f.write(classification_report(y_test_f, y_pred_f, zero_division=0))
    f.write("\nOverload Detection Report:\n")
    f.write(classification_report(y_test_o, y_pred_o, zero_division=0))

print(f"✅ All results saved to: {output_dir}")
