import joblib
import pandas as pd

# 경로 설정
MODEL_PATH = "models/rf_model.pkl"
DATA_PATH  = "data/normalized_data.csv"
TARGET     = "sleep_efficiency"

# 모델 불러오기
model = joblib.load(MODEL_PATH)

# 데이터 불러오기
df = pd.read_csv(DATA_PATH)
X = df[model.feature_names_in_]
y = df[TARGET]

# X + y를 하나로 합쳐서 저장
df_full = X.copy()
df_full["target"] = y
df_full.to_csv("X_and_y_data.csv", index=False)

# 피처 중요도 저장
importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
importance.sort_values(ascending=False).to_csv("feature_importance.csv", header=["importance"])

print(" X_and_y_data.csv / feature_importance.csv 저장 완료!")
