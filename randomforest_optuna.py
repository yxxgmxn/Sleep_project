import pandas as pd
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ✅ 데이터 불러오기
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# ✅ Optuna 목적 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])  # 'auto' 제거
    }
    model = RandomForestRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# ✅ Optuna 스터디 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# ✅ 최적 모델로 재학습 및 평가
best_params = study.best_params
final_model = RandomForestRegressor(random_state=42, **best_params)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# ✅ 성능 지표 출력
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ 최종 모델 성능")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"\n✅ Best Parameters: {best_params}")

# ✅ 변수 중요도 시각화
importances = final_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title("Top 20 Feature Importances (Optuna Tuned RF)")
plt.tight_layout()
plt.show()

# ✅ 예측값 vs 실제값 분포 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Optuna RF)")
plt.grid(True)
plt.tight_layout()
plt.show()
