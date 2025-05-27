import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import joblib

# ---------------- Load and Prepare Dataset ----------------
df = pd.read_csv("smart_grid_dataset.csv")
df_clean = df.drop(columns=["Timestamp"])

# Convert Fault Indicator to binary
df_clean["Binary Fault"] = df_clean["Fault Indicator"].apply(lambda x: 0 if x == 0 else 1)

# ---------------- Power Usage Prediction ----------------
X_reg = df_clean.drop(columns=["Power Usage (kW)", "Fault Indicator", "Binary Fault"])
y_reg = df_clean["Power Usage (kW)"]

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42)

sel_model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
sel_model_reg.fit(X_reg_train, y_reg_train)
selector_reg = SelectFromModel(sel_model_reg, prefit=True)
X_reg_train_sel = selector_reg.transform(X_reg_train)
X_reg_test_sel = selector_reg.transform(X_reg_test)

reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_reg_train_sel, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test_sel)

print("✅ Power Usage Prediction MSE:", mean_squared_error(y_reg_test, y_reg_pred))

# ---------------- Binary Fault Detection ----------------
X_clf = df_clean.drop(columns=["Power Usage (kW)", "Fault Indicator", "Binary Fault"])
y_clf = df_clean["Binary Fault"]

scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)

smote = SMOTE(random_state=42)
X_clf_bal, y_clf_bal = smote.fit_resample(X_clf_scaled, y_clf)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf_bal, y_clf_bal, test_size=0.2, random_state=42)

sel_model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
sel_model_clf.fit(X_clf_train, y_clf_train)
selector_clf = SelectFromModel(sel_model_clf, prefit=True)
X_clf_train_sel = selector_clf.transform(X_clf_train)
X_clf_test_sel = selector_clf.transform(X_clf_test)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_clf_train_sel, y_clf_train)
y_clf_pred = clf_model.predict(X_clf_test_sel)

print("\n✅ Binary Fault Detection Accuracy:", accuracy_score(y_clf_test, y_clf_pred))
print(classification_report(y_clf_test, y_clf_pred, target_names=["Normal", "Fault"]))

# ---------------- Confusion Matrix ----------------
conf_matrix = confusion_matrix(y_clf_test, y_clf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fault"], yticklabels=["Normal", "Fault"])
plt.title("Confusion Matrix - Binary Fault Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_binary_fault.png")
plt.close()

# ---------------- Regression Scatter Plot ----------------
plt.figure(figsize=(6, 5))
plt.scatter(y_reg_test, y_reg_pred, color='teal', alpha=0.6)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()],
         color='red', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted Power Usage (kW)")
plt.xlabel("Actual Power Usage")
plt.ylabel("Predicted Power Usage")
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_actual_vs_predicted.png")
plt.close()

# ---------------- Save All to Excel ----------------
with pd.ExcelWriter("smart_grid_predictions.xlsx") as writer:
    # Power predictions
    pd.DataFrame({
        "Actual Power Usage (kW)": y_reg_test.values,
        "Predicted Power Usage (kW)": y_reg_pred
    }).to_excel(writer, sheet_name="Power Usage Predictions", index=False)

    # Fault predictions
    pd.DataFrame({
        "Actual Fault (Binary)": y_clf_test,
        "Predicted Fault (Binary)": y_clf_pred
    }).to_excel(writer, sheet_name="Fault Predictions", index=False)

# ---------------- Save Trained Models ----------------
joblib.dump(reg_model, "regression_model_binary.pkl")
joblib.dump(clf_model, "classification_model_binary.pkl")

print("\n🎉 All models, predictions, and plots saved successfully.")
