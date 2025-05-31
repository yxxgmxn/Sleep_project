import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 모델 불러오기
model = joblib.load("models/rf_model.pkl")

print(f"모델 타입: {type(model).__name__}")
print(f"트리 수: {model.n_estimators}")
print(f"최대 깊이: {model.max_depth}")
print(f"입력 피처 수: {len(model.feature_names_in_)}")
print("주요 입력 피처:")
for i, name in enumerate(model.feature_names_in_):
    print(f"  {i+1:2d}. {name}")

# 변수 중요도 상위 10개 출력
importances = model.feature_importances_
top10 = sorted(zip(model.feature_names_in_, importances), key=lambda x: x[1], reverse=True)[:10]
print("\n상위 10개 중요 피처:")
for name, score in top10:
    print(f"{name:30s} importance={score:.4f}")

# 중요도 시각화 
pd.Series(dict(top10)).sort_values().plot.barh(title="Top 10 Feature Importances")
plt.tight_layout()
plt.show()
