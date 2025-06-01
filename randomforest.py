# randomforest.py

# 그래프 백엔드 설정 
import matplotlib
matplotlib.use('TkAgg')  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ 데이터 불러오기
print("📥 데이터 불러오는 중...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# ✅ 모델 생성 및 학습
print("🌲 Random Forest 모델 학습 중...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ 예측
print("🔍 예측 수행 중...")
y_pred = model.predict(X_test)

# ✅ 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n[✔] MSE       : {mse:.4f}")
print(f"[✔] R2 Score  : {r2:.4f}")

# ✅ 변수 중요도 시각화
print("📊 변수 중요도 시각화...")
importances = model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# ✅ 예측값 분포 시각화
print("📈 예측값 분포 시각화...")
plt.figure(figsize=(8, 6))
sns.histplot(y_pred, bins=100, kde=True)
plt.title("Distribution of Sleep Quality Score of predict")
plt.xlabel("Predicted Sleep Quality Score")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
