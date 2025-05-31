import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ✅ 데이터 불러오기
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# ✅ 랜덤 탐색을 위한 파라미터 그리드
param_distributions = {
    'n_estimators': np.arange(100, 501, 50),
    'max_depth': [None] + list(np.arange(10, 51, 10)),
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("🔍 RandomizedSearchCV 시작 중...")

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=30,
    scoring='r2'
    ,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print(f"\n✅ 최적 모델 파라미터: {random_search.best_params_}")

# ✅ 예측 및 성능 평가
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 예측 성능")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# ✅ 변수 중요도 시각화
importances = best_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title("Top 20 Feature Importances (RandomizedSearch RF)")
plt.tight_layout()
plt.show()
