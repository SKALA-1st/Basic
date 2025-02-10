# -------------------------------------------------------------------
# 작성자 : 저메추해조 (손지영, 김민주, 이재웅, 김다은)
# 작성목적 : 데이터 분석 실습 (ESG co2 순배출량 추이 예측 분석)
# 작성일 : 2025-02-10
#
# 변경사항 내역 (날짜, 변경목적, 변경내용 순으로 기입)
# - 예측 그래프 추가
# - 추세선 추가
# - R-squared, MSE 추가
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# ----- 파일 읽기 -----
file_path = 'ESG_CO2_2021.csv'  # 파일 경로를 현재 디렉토리로 변경
# 데이터 로드 (UTF-8 인코딩 시도 후 실패 시 CP949 시도)
try:
    df = pd.read_csv(file_path, encoding='utf-8')  # UTF-8 인코딩 시도
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp949')  # CP949 인코딩 시도
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='euc-kr')  # EUC-KR 인코딩 시도


# ---- 데이터 전처리 -----
# 결측값 확인
print(df.isnull().sum())

# 이상치 확인
# 1사분위수(Q1), 3사분위수(Q3) 계산
Q1 = df['Net'].quantile(0.25)  # 25th Percentile (1사분위수)
Q3 = df['Net'].quantile(0.75)  # 75th Percentile (3사분위수)
IQR = Q3 - Q1  # IQR 계산

# 하한선과 상한선
lower_bound = Q1 - 1.5 * IQR  # 하한선
upper_bound = Q3 + 1.5 * IQR  # 상한선

# 이상치 탐지
outliers = df[(df['Net'] < lower_bound) | (df['Net'] > upper_bound)]
print('이상치 없음')
print(outliers)


# ----- 예측 모델 생성 -----
# Scikit-learn으로 선형 회귀 구현
X=df[['year']]  # 독립 변수: 2D 배열 형태
y=df['Net']     # 종속 변수

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 향후 5년 데이터 예측
future_years = np.arange(2022, 2027).reshape(-1, 1)  # 2022년부터 2026년까지
future_predictions = model.predict(future_years)  # 예측 값 생성
y_pred = model.predict(X)


# ----- R-square, MSE 값 출력 -----

# R-squared 계산
r2 = r2_score(y, y_pred)

# MSE 계산
mae = mean_absolute_error(y, y_pred)

print('\n\n# ----- R-square, MSE 값 출력 -----')
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")


# ----- 회귀 계수와 절편 출력 -----
print('\n\n# ----- 그래프 그리기 -----')
print('회귀계수 & 절편:')
print(f"- 회귀 계수 (기울기): {model.coef_[0]:.4f}")
print(f"- 절편: {model.intercept_:.4f}")
print(f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")

# Y축 값에 쉼표 추가하는 함수
def format_y(value, tick_number):
    return f'{value:,.0f}'  # 쉼표 추가, 소수점 0자리

# 그래프 그리기
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='year', y='Net', color='blue', marker='o', label='Actual Data') # 실제 데이터 (파란색 꺾은선 그래프)
plt.plot(df['year'], y_pred, color='green', linestyle='--', label='Linear Regression') # 회귀선 (빨간색 선)
plt.plot(future_years, future_predictions, marker='o', color='red', label='Future Predictions') # 예측 데이터(초록색 선)
plt.ylim(0, df['Net'].max() * 1.25)  # Net의 최댓값의 1.05%를 y축의 최대값으로 설정
plt.xticks(range(df['year'].min(), df['year'].max()+5, 5))
plt.fill_between(future_years.flatten(), future_predictions, alpha=0.2,color='red',label='Prediction Range')
y_min = max(0, df['Net'].min() - 100000)  # 최소값이 음수면 0으로 조정
y_max = df['Net'].max() * 1.25  # Net의 최댓값의 1.25%를 y축의 최대값으로 설정
plt.ylim(y_min, y_max)   # y축 범위 설정

# 예측값 데이터 라벨 표시
for i, txt in enumerate(future_predictions):
    plt.annotate(f"{format_y(txt, None)}", (future_years[i], future_predictions[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10, color="black", rotation=60)

# 그래프 제목 및 라벨
plt.title('ESG CO2 Net Emissions Trend Analysis', fontsize=16)
plt.xlabel('year', fontsize=12)
plt.ylabel('Net', fontsize=12)
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y)) # y축 쉼표 추가

# 그래프 추가 꾸미기
plt.grid(True, linestyle='--', alpha=0.7) # 격자눈
plt.legend() # 범례
plt.tight_layout()


# ----- 그래프 출력 -----

# 그래프 이미지로 저장
image_file_path = "co2_emissions_prediction_graph.png"
plt.savefig(image_file_path)
print('\n\n# ----- 그래프 저장 -----')
print(f"그래프가 '{image_file_path}'로 저장되었습니다.")

# 그래프 출력
print('\n\n# ----- 그래프 출력 -----')
plt.show()