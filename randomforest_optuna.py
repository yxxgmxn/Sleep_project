import pandas as pd
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
    }
    model = RandomForestRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# ✅ Optuna 튜닝 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# ✅ 결과 출력
print("✅ Best R2 Score:", study.best_value)
print("✅ Best Parameters:")
for key, val in study.best_params.items():
    print(f" - {key}: {val}")

# ✅ 최적 파라미터로 모델 재학습
best_model = RandomForestRegressor(random_state=42, **study.best_params)
best_model.fit(X_train, y_train)

# ✅ 변수 중요도 시각화
importances = best_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title("Top 20 Feature Importances (Optuna Tuned RF)")
plt.tight_layout()
plt.show()
