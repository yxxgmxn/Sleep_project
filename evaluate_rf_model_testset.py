import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 경로 설정
MODEL_PATH = "models/rf_model.pkl"
DATA_PATH = "data/normalized_data.csv"
TARGET = "sleep_efficiency"
SEED = 42

# ── 모델 및 데이터 로드
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ── 입력/타겟 분리
X = df[model.feature_names_in_]
y = df[TARGET]

# ── 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ── 예측 수행
y_pred = model.predict(X_test)

# ── 성능 평가
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

# ── 결과 출력
print("테스트셋 기준 모델 일반화 성능:")
print(f"R² (설명력): {r2:.4f}")
print(f"MAE (평균 절대 오차): {mae:.4f}")
print(f"RMSE (평균 제곱근 오차): {rmse:.4f}")

# ── 결과 저장
results_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
results_df.to_csv("output/rf_testset_predictions.csv", index=False)
print("예측 결과 저장 완료: output/rf_testset_predictions.csv")
