import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import joblib

def run_model10():
    # === Load and preprocess data ===
    df = pd.read_csv("smart_grid_dataset1.csv")

    # Fix timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d-%m-%Y %H.%M")

    # Sort and reset index
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # === Feature Engineering ===
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Heating_Degree_Days'] = np.maximum(0, 18 - df['Temperature_C'])
    df['Cooling_Degree_Days'] = np.maximum(0, df['Temperature_C'] - 18)
    df['Net_Grid_Usage'] = df['Grid_Supply(kW)'] - (df['Solar_Power(kW)'] + df['Wind_Power(kW)'])
    df['Price_Elasticity'] = df['Electricity_Price(USD/kWh)'] * df['Power_Consumption(kW)']
    df['Solar_Wind_Ratio'] = df['Solar_Power(kW)'] / (df['Wind_Power(kW)'] + 1e-3)
    df['Humidity_Temp_Interaction'] = df['Humidity(%)'] * df['Temperature_C']
    df['Temperature_Sq'] = df['Temperature_C'] ** 2
    df['Power_Lag1'] = df['Power_Consumption(kW)'].shift(1)
    df['Power_Lag2'] = df['Power_Consumption(kW)'].shift(2)
    df['Power_Lag3'] = df['Power_Consumption(kW)'].shift(3)

    df.dropna(inplace=True)

    # === Define features and target ===
    features = [
        'Voltage(V)', 'Current(A)', 'Reactive_Power(kVAR)', 'Power_Factor',
        'Cooling_Degree_Days', 'Net_Grid_Usage', 'Price_Elasticity', 'Electricity_Price(USD/kWh)',
        'Temperature_Sq', 'Solar_Wind_Ratio', 'Humidity_Temp_Interaction',
        'Heating_Degree_Days', 'Power_Lag3', 'Hour', 'Power_Lag2', 'DayOfWeek'
    ]
    target = 'Power_Consumption(kW)'

    X = df[features]
    y = df[target]

    # === Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Train XGBoost ===
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        early_stopping_rounds=50,
        verbosity=1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    joblib.dump(model, "xgb_model.joblib")
    print("✅ Model saved to xgb_model.joblib")

    # === Predictions and Evaluation ===
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # === Output Directory Setup ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("model10_outputs", f"output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # === Save Predictions (Actual vs Predicted only) ===
    result_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    result_df.to_csv(os.path.join(output_dir, f"power_prediction_{timestamp}.csv"), index=False)

    # === Power Plot ===
    plt.figure(figsize=(12, 5))
    plt.plot(result_df['Actual'].values, label='Actual', alpha=0.7)
    plt.plot(result_df['Predicted'].values, label='Predicted', alpha=0.7)
    plt.title("Power Consumption: Actual vs Predicted")
    plt.xlabel("Samples")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"power_plot_{timestamp}.png"))
    plt.close()

    # === Anomaly Detection using One-Class SVM ===
    scaler = StandardScaler()
    fault_features = ['Voltage(V)', 'Current(A)', 'Reactive_Power(kVAR)', 'Voltage_Fluctuation(%)']

    df_scaled = scaler.fit_transform(df[fault_features])

    for label in ['Overload', 'Transformer_Fault']:
        svm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
        svm.fit(df_scaled)
        preds = svm.predict(df_scaled)
        preds = np.where(preds == -1, 1, 0)
        cm = confusion_matrix(df[label], preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {label}")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{label}_{timestamp}.png"))
        plt.close()
        df[f'Anomaly_{label}'] = preds

    # Save anomaly results
    anomaly_cols = ['Timestamp', 'Overload', 'Anomaly_Overload', 'Transformer_Fault', 'Anomaly_Transformer_Fault']
    df[anomaly_cols].to_csv(os.path.join(output_dir, f"anomaly_detection_{timestamp}.csv"), index=False)

    # === Print Summary ===
    print("\n📊 Final XGBoost Evaluation:")
    print(f"🔹 MSE: {mse:.4f}")
    print(f"🔹 RMSE: {rmse:.4f}")
    print(f"🔹 R² Score: {r2:.4f}")
    print(f"✅ Power predictions saved to: {os.path.join(output_dir, f'power_prediction_{timestamp}.csv')}")
    print(f"📈 Power plot saved to: {os.path.join(output_dir, f'power_plot_{timestamp}.png')}")
    print(f"🧩 Confusion matrices saved in: {output_dir}/")
    print(f"🛠️ Anomaly results saved to: {os.path.join(output_dir, f'anomaly_detection_{timestamp}.csv')}")

def feature_engineer_dataframe(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d-%m-%Y %H.%M")
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Heating_Degree_Days'] = np.maximum(0, 18 - df['Temperature_C'])
    df['Cooling_Degree_Days'] = np.maximum(0, df['Temperature_C'] - 18)
    df['Net_Grid_Usage'] = df['Grid_Supply(kW)'] - (df['Solar_Power(kW)'] + df['Wind_Power(kW)'])
    df['Price_Elasticity'] = df['Electricity_Price(USD/kWh)'] * df['Power_Consumption(kW)']
    df['Solar_Wind_Ratio'] = df['Solar_Power(kW)'] / (df['Wind_Power(kW)'] + 1e-3)
    df['Humidity_Temp_Interaction'] = df['Humidity(%)'] * df['Temperature_C']
    df['Temperature_Sq'] = df['Temperature_C'] ** 2
    df['Power_Lag1'] = df['Power_Consumption(kW)'].shift(1)
    df['Power_Lag2'] = df['Power_Consumption(kW)'].shift(2)
    df['Power_Lag3'] = df['Power_Consumption(kW)'].shift(3)
    df.dropna(inplace=True)
    return df


def predict_from_csv(input_csv_path, model_path, output_csv_path=None):
    import os
    from datetime import datetime
    import matplotlib.pyplot as plt
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = pd.read_csv(input_csv_path)
    df = feature_engineer_dataframe(df)

    features = [
        'Voltage(V)', 'Current(A)', 'Reactive_Power(kVAR)', 'Power_Factor',
        'Cooling_Degree_Days', 'Net_Grid_Usage', 'Price_Elasticity', 'Electricity_Price(USD/kWh)',
        'Temperature_Sq', 'Solar_Wind_Ratio', 'Humidity_Temp_Interaction',
        'Heating_Degree_Days', 'Power_Lag3', 'Hour', 'Power_Lag2', 'DayOfWeek'
    ]

    model = joblib.load(model_path)
    df['Predicted_Power_Consumption(kW)'] = model.predict(df[features])

    # === Create output directory ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("prediction_outputs", f"output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # === Save Predictions CSV ===
    pred_csv_path = os.path.join(output_dir, "power_predictions.csv")
    df[['Timestamp', 'Predicted_Power_Consumption(kW)']].to_csv(pred_csv_path, index=False)

    # === Save Prediction Plot ===
    plt.figure(figsize=(12, 5))
    plt.plot(df['Predicted_Power_Consumption(kW)'], label='Predicted', color='blue')
    plt.title("Predicted Power Consumption")
    plt.xlabel("Samples")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_plot.png"))
    plt.close()

    # === Anomaly Detection ===
    fault_features = ['Voltage(V)', 'Current(A)', 'Reactive_Power(kVAR)', 'Voltage_Fluctuation(%)']
    if all(col in df.columns for col in fault_features):
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[fault_features])

        for label in ['Overload', 'Transformer_Fault']:
            if label in df.columns:
                svm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
                svm.fit(df_scaled)
                preds = svm.predict(df_scaled)
                preds = np.where(preds == -1, 1, 0)
                df[f'Anomaly_{label}'] = preds

                cm = confusion_matrix(df[label], preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.title(f"Confusion Matrix - {label}")
                plt.savefig(os.path.join(output_dir, f"confusion_matrix_{label}.png"))
                plt.close()

        # Save anomaly results
        anomaly_cols = ['Timestamp', 'Overload', 'Anomaly_Overload', 'Transformer_Fault', 'Anomaly_Transformer_Fault']
        df[[col for col in anomaly_cols if col in df.columns]].to_csv(
            os.path.join(output_dir, "anomaly_results.csv"), index=False
        )

    print(f"✅ All outputs saved in: {output_dir}")

if __name__ == "__main__":
    run_model10()
