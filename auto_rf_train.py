
from pathlib import Path
import pandas as pd, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 경로 설정 
BASE   = Path(__file__).resolve().parent
DATA   = BASE / "data"   / "normalized_data.csv"
MODEL  = BASE / "models" / "rf_model.pkl"
OUTPUT = BASE / "output"

# ── 하이퍼파라미터 및 반복 설정 
TARGET      = "sleep_efficiency"  # 예측 타깃 컬럼
DROP_COLS   = [c for c in ("timestamp", "patient_id") if c in pd.read_csv(DATA, nrows=1).columns]

INCREMENT   = 20     # 트리 수
MAX_ROUNDS  = 5000     #반복 횟수
MAX_DEPTH   = 10     # 트리 최대 깊이
THRESHOLD   = 0 # R² 개선폭이 이 값보다 작으면 멈춤
SEED        = 42

# ── 데이터 로드 
df = pd.read_csv(DATA)
X  = df.drop(columns=[TARGET] + DROP_COLS)
y  = df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ── 새 모델 생성 
rf = RandomForestRegressor(
    n_estimators=0,
    warm_start=True,
    max_depth=MAX_DEPTH,
    n_jobs=-1,
    random_state=SEED,
)

best_r2 = -1.0
for rnd in range(1, MAX_ROUNDS + 1):
    rf.set_params(n_estimators=rf.n_estimators + INCREMENT)
    rf.fit(X_tr, y_tr)

    pred = rf.predict(X_te)
    r2   = r2_score(y_te, pred)
    mae  = mean_absolute_error(y_te, pred)
    mse  = mean_squared_error(y_te, pred)   # squared 인자 없이 MSE 계산
    rmse = mse ** 0.5                       # √MSE = RMSE

    print(f"[round {rnd:02d}] trees={rf.n_estimators:4d} | "
      f"ΔR²={r2 - best_r2:.6f} | R²={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")


   
    # 조기 종료 안 함 → THRESHOLD = 0
    if r2 - best_r2 < THRESHOLD:
        pass
    best_r2 = r2

# ── 모델·결과 저장 
MODEL.parent.mkdir(exist_ok=True)
joblib.dump(rf, MODEL)
print("최종 모델 저장 →", MODEL)

OUTPUT.mkdir(exist_ok=True)
pd.DataFrame({"y_true": y_te, "y_pred": pred}).to_csv(
    OUTPUT / "rf_predictions.csv", index=False
)

# 변수 중요도 상위 25개 시각화
imp = (
    pd.Series(rf.feature_importances_, index=X.columns)
    .sort_values(ascending=False)[:25]
)
plt.figure(figsize=(8, 6))
imp.plot.barh()
plt.gca().invert_yaxis()
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig(OUTPUT / "feature_importance.png")
plt.close()
print("feature_importance.png 저장 완료")
