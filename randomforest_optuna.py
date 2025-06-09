import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 데이터 불러오기 및 표준화
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# sleep_quality_score 출력
print('sleep_quality_score 최소값:', y_train.min())
print('sleep_quality_score 최대값:', y_train.max())
print(pd.Series(y_train).describe())

# ✅ Optuna 목적 함수 (5-fold CV 적용)
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }
    model = RandomForestRegressor(random_state=42, **params, n_jobs=-1)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    return np.mean(scores)

# ✅ 10개마다 진행상황 출력 콜백 함수
def print_every_10_trials(study, trial):
    if (trial.number + 1) % 10 == 0:
        print(f"\n===== {trial.number + 1}개 trial 완료! =====")
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_params}")

# ✅ Optuna Study 실행 
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(
    objective,
    n_trials=100,
    n_jobs=12,
    callbacks=[print_every_10_trials]
)

# ✅ 최적 모델 학습 및 평가
best_params = study.best_params
print("\n✅ Best Parameters:", best_params)

final_model = RandomForestRegressor(random_state=42, **best_params, n_jobs=-1)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)

# ✅ 예측값 보정 (0~100 범위로 clip)
y_pred = np.clip(y_pred, 0, 100)

# ✅ 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ 최종 모델 성능")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# ✅ 예측 결과 저장
pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv("rf_predictions.csv", index=False)
print("📄 예측 결과 저장 완료: rf_predictions.csv")

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

# ✅ 예측값 vs 실제값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Optuna RF)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 예측 결과 3개 구간 분류 (bad/mid/good)
bins = np.quantile(y_test, [0, 0.33, 0.66, 1.0])
y_test_cls = np.digitize(y_test, bins=bins[1:], right=False)
y_pred_cls = np.digitize(y_pred, bins=bins[1:], right=False)

from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test_cls, y_pred_cls)
report = classification_report(y_test_cls, y_pred_cls, target_names=['bad', 'mid', 'good'])

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred_bad', 'Pred_mid', 'Pred_good'],
            yticklabels=['True_bad', 'True_mid', 'True_good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (3-class by quantile)')
plt.tight_layout()
plt.show()
