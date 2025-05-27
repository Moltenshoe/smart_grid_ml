import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor

# --- Config ---
MODEL_NAME = "ai_smart_grid_model5_auto_threshold"
RANDOM_STATE = 42
FALLBACK_THRESHOLD = 0.75

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

# --- Normalize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split ---
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_scaled, y_power, test_size=0.2, random_state=RANDOM_STATE)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_scaled, y_overload, test_size=0.2, random_state=RANDOM_STATE)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled, y_fault, test_size=0.2, random_state=RANDOM_STATE)

# --- Train Fault Detection Model ---
print("Training Fault Detection Model with smart F1 thresholding...")
pos_weight = len(y_train_f[y_train_f == 0]) / len(y_train_f[y_train_f == 1])
model_fault = GridSearchCV(
    XGBClassifier(eval_metric='logloss', subsample=0.8, random_state=RANDOM_STATE),
    param_grid={
        'max_depth': [3],
        'min_child_weight': [6],
        'gamma': [0.3],
        'scale_pos_weight': [round(pos_weight)]
    },
    cv=3, scoring='f1', n_jobs=-1
)
model_fault.fit(X_train_f, y_train_f)
y_proba_f = model_fault.predict_proba(X_test_f)[:, 1]

# --- Smart threshold search ---
best_threshold = None
for threshold in np.linspace(0.99, 0.01, 100):
    preds = (y_proba_f > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_f, preds).ravel()
    if fp <= 400 and fn <= 100:
        best_threshold = threshold
        break

if best_threshold is None:
    print(f"\n⚠️ No threshold found meeting FP≤400 and FN≤100. Using fallback = {FALLBACK_THRESHOLD}")
    best_threshold = FALLBACK_THRESHOLD
else:
    print(f"\n✅ Best Threshold Found: {best_threshold:.4f} (FP ≤ 400, FN ≤ 100)")

y_pred_f = (y_proba_f > best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test_f, y_pred_f).ravel()
print("\n📊 Fault Confusion Matrix:")
print(f"[[{tn} {fp}]\n [{fn} {tp}]]")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
print(f"True Negatives:  {tn}")

# --- Overload Detection ---
print("Training Overload Detection Model...")
model_overload = XGBClassifier(max_depth=3, min_child_weight=2, random_state=RANDOM_STATE)
model_overload.fit(X_train_o, y_train_o)
y_pred_o = model_overload.predict(X_test_o)

# --- Power Prediction ---
print("Training Power Usage Prediction Model...")
model_power = XGBRegressor(max_depth=4, random_state=RANDOM_STATE)
model_power.fit(X_train_p, y_train_p)
y_pred_p = model_power.predict(X_test_p)

# --- Output Directory ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"{MODEL_NAME}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# --- Save predictions
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

# --- Confusion Matrices
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

# --- PR Curve
precision, recall, thresholds = precision_recall_curve(y_test_f, y_proba_f)
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

# --- Histogram
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

# --- Performance Report
with open(os.path.join(output_dir, "performance_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Threshold Used: {best_threshold:.4f} (Fallback: {FALLBACK_THRESHOLD})\n")
    f.write(f"Best Fault Model Parameters: {model_fault.best_params_}\n")
    f.write(f"Power Prediction MSE: {mean_squared_error(y_test_p, y_pred_p):.6f}\n\n")
    f.write("Fault Detection Report:\n")
    f.write(classification_report(y_test_f, y_pred_f, zero_division=0))
    f.write("\nOverload Detection Report:\n")
    f.write(classification_report(y_test_o, y_pred_o, zero_division=0))

print(f"\n✅ All results saved to: {output_dir}")
