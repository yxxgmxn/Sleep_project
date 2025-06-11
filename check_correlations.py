import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── 1. 데이터 로드
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

df = X_train.copy()
df["sleep_quality_score"] = y_train.values

# ── 2. 상관계수 계산 
corr = df.corr(numeric_only=True)

# sleep_quality_score와의 상위 10개 추출
target_corr = corr["sleep_quality_score"].drop("sleep_quality_score")
top25 = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)[:10]

# ── 3. 콘솔·CSV 출력 
print("\n📑 상위 10개 상관계수")
print(top25)
top25.to_csv("corr_top25.csv", header=["correlation"])
print("✅ corr_top25.csv 저장 완료")

# ── 4. 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(
    df[top25.index.tolist() + ["sleep_quality_score"]].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, center=0
)
plt.title("Correlation Heatmap (Top 25 Features vs sleep_quality_score)")
plt.tight_layout()
plt.show()
