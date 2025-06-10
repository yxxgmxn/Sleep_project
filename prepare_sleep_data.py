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
    return score * 100

# 👉 원시 점수 계산 후 정규화
data['raw_score'] = data.apply(compute_sleep_quality_score, axis=1)
min_score = data['raw_score'].min()
max_score = data['raw_score'].max()
data['sleep_quality_score'] = 100 * (data['raw_score'] - min_score) / (max_score - min_score)
data['sleep_quality_score'] = data['sleep_quality_score'].round(2)
data.drop(columns=['raw_score'], inplace=True)

print("\n📈 sleep_quality_score 통계:")
print(data['sleep_quality_score'].describe())

# 4. 학습에 사용하지 않을 컬럼 삭제 (팀원 기준)
remove_cols = [
    'sleep_length', 'sleep_efficiency', 'sleep_onset_dev', 'sleep_counts',
    'sleep_offset_dev', 'late_sleep_offset',
    'sleep_length_ndays_mean', 'sleep_length_ndays_stdev', 'sleep_length_ndays_gradient',
    'sleep_efficiency_ndays_mean', 'sleep_efficiency_ndays_stdev', 'sleep_efficiency_ndays_gradient',
    'sleep_onset_dev_ndays_mean', 'sleep_onset_dev_ndays_stdev', 'sleep_onset_dev_ndays_gradient',
    'sleep_offset_dev_ndays_mean', 'sleep_offset_dev_ndays_stdev', 'sleep_offset_dev_ndays_gradient',
    'sleep_counts_ndays_mean', 'sleep_counts_ndays_stdev', 'sleep_counts_ndays_gradient',
    'late_sleep_offset_ndays_mean', 'late_sleep_offset_ndays_stdev', 'late_sleep_offset_ndays_gradient',
]
data = data.drop(columns=[col for col in remove_cols if col in data.columns])

# 5. 피처/타깃 분리
drop_cols = ['sleep_quality_score', 'score_level', 'patient_id', 'timestamp']
drop_cols = [col for col in drop_cols if col in data.columns]
X = data.drop(columns=drop_cols)
y = data['sleep_quality_score']

# 6. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✅ 데이터 분할 완료: X_train={X_train.shape}, X_test={X_test.shape}")

# 7. 저장
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("📁 파일 저장 완료: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
