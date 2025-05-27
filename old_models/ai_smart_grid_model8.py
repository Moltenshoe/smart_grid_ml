import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Setup Output Directory -----------
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{script_name}_outputs_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# ----------- Load Dataset -----------
df = pd.read_csv('smart_grid_dataset1.csv')

# ----------- Validate Columns -----------
required_columns = ['Transformer Fault', 'Overload Condition', 'Timestamp', 'Power Consumption (kW)']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

# ----------- Utility: Save Confusion Matrix -----------
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# ----------- Transformer Fault Detection -----------
X_fault = df.drop(columns=['Transformer Fault', 'Overload Condition', 'Timestamp', 'Power Consumption (kW)'])
y_fault = df['Transformer Fault']
X_fault_train, X_fault_test, y_fault_train, y_fault_test = train_test_split(X_fault, y_fault, test_size=0.3, random_state=42)
scaler_fault = StandardScaler()
X_fault_train_scaled = scaler_fault.fit_transform(X_fault_train)
X_fault_test_scaled = scaler_fault.transform(X_fault_test)

smote_tomek_fault = SMOTETomek(random_state=42)
X_fault_res, y_fault_res = smote_tomek_fault.fit_resample(X_fault_train_scaled, y_fault_train)

fault_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
fault_model.fit(X_fault_res, y_fault_res)
y_fault_pred = fault_model.predict(X_fault_test_scaled)

# Save fault results
plot_confusion_matrix(y_fault_test, y_fault_pred, "Transformer Fault Confusion Matrix", "conf_matrix_fault.png")
fault_report = classification_report(y_fault_test, y_fault_pred, output_dict=True)
pd.DataFrame(fault_report).transpose().to_excel(os.path.join(output_dir, "fault_classification_report.xlsx"))

# ----------- Overload Condition Detection -----------
X_overload = df.drop(columns=['Overload Condition', 'Transformer Fault', 'Timestamp', 'Power Consumption (kW)'])
y_overload = df['Overload Condition']
X_ov_train, X_ov_test, y_ov_train, y_ov_test = train_test_split(X_overload, y_overload, test_size=0.3, random_state=42)
scaler_ov = StandardScaler()
X_ov_train_scaled = scaler_ov.fit_transform(X_ov_train)
X_ov_test_scaled = scaler_ov.transform(X_ov_test)

smote_tomek_ov = SMOTETomek(random_state=42)
X_ov_res, y_ov_res = smote_tomek_ov.fit_resample(X_ov_train_scaled, y_ov_train)

ov_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
ov_model.fit(X_ov_res, y_ov_res)
y_ov_pred = ov_model.predict(X_ov_test_scaled)

# Save overload results
plot_confusion_matrix(y_ov_test, y_ov_pred, "Overload Condition Confusion Matrix", "conf_matrix_overload.png")
ov_report = classification_report(y_ov_test, y_ov_pred, output_dict=True)
pd.DataFrame(ov_report).transpose().to_excel(os.path.join(output_dir, "overload_classification_report.xlsx"))

# ----------- Power Consumption Prediction -----------
X_power = df.drop(columns=['Power Consumption (kW)', 'Transformer Fault', 'Overload Condition', 'Timestamp'])
y_power = df['Power Consumption (kW)']
X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_power, y_power, test_size=0.3, random_state=42)
scaler_p = StandardScaler()
X_p_train_scaled = scaler_p.fit_transform(X_p_train)
X_p_test_scaled = scaler_p.transform(X_p_test)

power_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
power_model.fit(X_p_train_scaled, y_p_train)
y_p_pred = power_model.predict(X_p_test_scaled)

# Save power prediction results
power_df = pd.DataFrame({'Actual Power Usage': y_p_test, 'Predicted Power Usage': y_p_pred})
power_df.to_excel(os.path.join(output_dir, "power_prediction.xlsx"), index=False)

mse = mean_squared_error(y_p_test, y_p_pred)
r2 = r2_score(y_p_test, y_p_pred)
with open(os.path.join(output_dir, "power_metrics.txt"), 'w') as f:
    f.write(f"Mean Squared Error: {mse:.2f}\n")
    f.write(f"R^2 Score: {r2:.2f}\n")

print(f"\n✅ All results saved in: {output_dir}")
