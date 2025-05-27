from ai_smart_grid_model10 import predict_from_csv

print("🔌 Smart Grid Power Predictor")
input_csv = input("📂 Enter path to the new input CSV file: ").strip()
model_path = input("🧠 Enter path to the trained XGBoost model (.joblib): ").strip()

predict_from_csv(input_csv, model_path)
