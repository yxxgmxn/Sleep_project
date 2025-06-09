import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
data = pd.read_csv('data.csv')

# 2. 수치형 컬럼 정규화 (patient_id 제외)
numeric_columns = data.select_dtypes(include=[np.number]).columns
numeric_columns = [col for col in numeric_columns if col != 'patient_id']
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

print("✅ 정규화가 완료되었습니다.\n")
print("📊 처음 5개 행:")
print(data.head())

# 3. 수면 질 점수 계산 및 0~100 정규화
def compute_sleep_quality_score(row):
    efficiency = row["sleep_efficiency"]
    length = row["sleep_length"]
    counts = row["sleep_counts"]
    onset_dev = row["sleep_onset_dev"]
    score = (0.35 * efficiency + 0.2 * length - 0.2 * counts - 0.25 * onset_dev)
    return score * 100  # ⛔️ round 제거!

# 👉 원시 점수 계산 후 정규화
data['raw_score'] = data.apply(compute_sleep_quality_score, axis=1)
min_score = data['raw_score'].min()
max_score = data['raw_score'].max()
data['sleep_quality_score'] = 100 * (data['raw_score'] - min_score) / (max_score - min_score)
data['sleep_quality_score'] = data['sleep_quality_score'].round(2)  # 🎯 마지막에만 반올림

# 👉 raw_score는 제거
data = data.drop(columns=['raw_score'])

print("\n📈 sleep_quality_score 통계:")
print(data['sleep_quality_score'].describe())

# 4. 원래 점수 구성요소 삭제
data = data.drop([
    'sleep_length',
    'sleep_efficiency',
    'sleep_onset_dev',
    'sleep_counts'
], axis=1)

# 5. 피처/타깃 분리
drop_cols = ['sleep_quality_score', 'score_level', 'patient_id', 'timestamp']
drop_cols = [col for col in drop_cols if col in data.columns]

features = data.drop(columns=drop_cols)
target = data['sleep_quality_score']

# 6. 학습/테스트 분할
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=random_seed
)

print(f"\n✅ 데이터 분할 완료: X_train={X_train.shape}, X_test={X_test.shape}")

# 7. 저장
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("📁 파일 저장 완료: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
