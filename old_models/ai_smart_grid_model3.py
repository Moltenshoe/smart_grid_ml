import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor

# --- Constants ---
MODEL_NAME = "ai_smart_grid_model3"
RANDOM_STATE = 42

# Load dataset
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

print("Original Class Distribution:")
print("Fault:", np.bincount(y_fault))
print("Overload:", np.bincount(y_overload))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_scaled, y_power, test_size=0.2, random_state=RANDOM_STATE)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_scaled, y_overload, test_size=0.2, random_state=RANDOM_STATE)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled, y_fault, test_size=0.2, random_state=RANDOM_STATE)

# Fault Detection with Conservative Settings
print("\nTraining Fault Detection Model (Precision-Optimized)...")
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)
X_train_f_bal, y_train_f_bal = smote.fit_resample(X_train_f, y_train_f)

param_grid = {
    'max_depth': [3, 4],
    'min_child_weight': [3, 5],
    'gamma': [0.1, 0.2],
    'scale_pos_weight': [5, 7, 10]  # Higher = more conservative
}

model_fault = GridSearchCV(
    XGBClassifier(eval_metric='logloss', subsample=0.8, random_state=RANDOM_STATE),
    param_grid,
    cv=3,
    scoring='precision',
    n_jobs=-1
)
model_fault.fit(X_train_f_bal, y_train_f_bal)
y_proba_f = model_fault.predict_proba(X_test_f)[:, 1]

# Select threshold where precision >= 0.90
precision, recall, thresholds = precision_recall_curve(y_test_f, y_proba_f)
target_precision = 0.90
optimal_threshold = 0.5  # default fallback
for p, t in zip(precision, thresholds):
    if p >= target_precision:
        optimal_threshold = t
        break

y_pred_f = (y_proba_f > optimal_threshold).astype(int)

# Overload Detection
print("Training Overload Detection Model...")
model_overload = XGBClassifier(max_depth=3, min_child_weight=2, random_state=RANDOM_STATE)
model_overload.fit(X_train_o, y_train_o)
y_pred_o = model_overload.predict(X_test_o)

# Power Prediction
print("Training Power Prediction Model...")
model_power = XGBRegressor(max_depth=4, random_state=RANDOM_STATE)
model_power.fit(X_train_p, y_train_p)
y_pred_p = model_power.predict(X_test_p)

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("results", f"{MODEL_NAME}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Save predictions to Excel
results_df = pd.DataFrame({
    'Actual_Power': y_test_p,
    'Predicted_Power': y_pred_p,
    'Actual_Overload': y_test_o,
    'Predicted_Overload': y_pred_o,
    'Actual_Fault': y_test_f,
    'Predicted_Fault': y_pred_f,
    'Fault_Probability': y_proba_f
})
results_df.to_excel(os.path.join(output_dir, "all_predictions.xlsx"), index=False)

# Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, title, filename, threshold=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Normal", "Predicted Fault"],
                yticklabels=["Actual Normal", "Actual Fault"],
                cbar=False)
    subtitle = f"\nThreshold: {threshold:.3f}" if threshold else ""
    plt.title(title + subtitle)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

plot_confusion_matrix(y_test_f, y_pred_f, "Fault Detection", "fault_confusion_matrix.png", optimal_threshold)
plot_confusion_matrix(y_test_o, y_pred_o, "Overload Detection", "overload_confusion_matrix.png")

# PR Curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve\n(Fault Detection, Threshold: {optimal_threshold:.3f})')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
plt.close()

# Report
with open(os.path.join(output_dir, "performance_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Optimal Threshold (Precision ≥ 0.90): {optimal_threshold:.4f}\n")
    f.write(f"Best Fault Model Params: {model_fault.best_params_}\n")
    f.write(f"Power Prediction MSE: {mean_squared_error(y_test_p, y_pred_p):.6f}\n\n")
    f.write("Fault Detection Report:\n")
    f.write(classification_report(y_test_f, y_pred_f, zero_division=0, target_names=["Normal", "Fault"]))
    f.write("\nOverload Detection Report:\n")
    f.write(classification_report(y_test_o, y_pred_o, zero_division=0, target_names=["Normal", "Overload"]))

# Console output
print(f"\n✅ All results saved to: {output_dir}")
print(f"Optimal Fault Threshold (precision ≥ 0.90): {optimal_threshold:.4f}")
print("Best Model Parameters (Fault):", model_fault.best_params_)
print("\n📊 Fault Detection Classification Report:")
print(classification_report(y_test_f, y_pred_f, target_names=["Normal", "Fault"]))
