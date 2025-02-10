# ----------------------------------------------------------------------------
#
# 작성자 : 가믿조 - 조장: 하가영, 임유나, 류지혜, 김현준
# 작성목적 : KDT 교육용 파이썬 pandas, numpy, seaborn, sklearn 실습 목적 코드
# 작성일 : 2025-02-08

# 본 파일은 KDT 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역 (날짜, 변경목적, 변경 내용 순으로 기입)
#  - 이상치 박스플롯 추가 
#  - 저장 이미지 확장자 추가 
#  - 사용자 파일 저장 확인 msg 추가 
#  - 향후 5년 예측치 추가 
#
# ----------------------------------------------------------------------------

# 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 파일 경로 설정
file_path = '/config/workspace/basic/2주차_1일차_데이터분석/ESG_CO2_2021.csv' # 파일 경로를 현재 디렉토리로 변경

# 데이터 로드 (UTF-8 인코딩 시도 후 실패시 CP949 시도)
try :
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError :
    try :
        df = pd.read_csv(file_path,encoding='cp949')
    except UnboundLocalError :
        df = pd.read_csv(file_path, encoding='euc-kr') # 다른 인코딩 시도

# 로드된 데이터 기본 정보 확인
print("-"*50)
print('데이터 타입 및 결측치 확인')
df.info()

print("-"*50)
print('상위 5개행 출력')
print(df.head())

print("-"*50)
print('요약 통계량 출력 (수치형 컬럼만)')
df.describe()

# 데이터 정제 및 전처리 

# 결측치 확인 및 처리 
print(df.isnull().sum()) # 각 컬럼의 결측치 개수 확인 

# 만약 결측치가 있다면 
# df = df.dropna()  # 결측치 삭제
# df['year'] = df['year'].fillna(df['year'].mean()) # 결측치 대체

# --> 결측치 없음

# 이상치 확인
Q1 = df['Net'].quantile(0.25) # 1사분위수
Q3 = df['Net'].quantile(0.75) # 3사부위수
IQR = Q3 - Q1 # IQR 계산

lower_bound = Q1 - 1.5*IQR # 하한선
upper_bound = Q3 + 1.5*IQR # 상한선

outliers = df[(df['Net'] < lower_bound) | (df['Net'] > upper_bound)] # 이상치 추출
print(outliers) 

# 이상치 시각화 (박스플롯)
plt.figure(figsize=(5, 5))
sns.boxplot(y=df['Net'])
plt.title('Boxplot of Net CO2')
plt.show()

# --> 이상치 없음

# 데이터 시각화 

x = df['year']
y = df['Net']

plt.plot(x,y,color='lightblue', label='Net(C02)') # 선 그래프
plt.legend() # 범례 작성

plt.scatter(x, y, color='lightblue', marker='.', label='Data Points') # 꺾은 점 추가

plt.grid(True) # 그리드 추가
plt.title('ESG CO2 Graph') # 제목 설정
plt.xlabel('year') # X축 레이블
plt.ylabel('Net') # Y축 레이블

# 그래프 이미지 변환 (plt.savefig(image_file_path))
img_path = 'ESG_graph_img.png'
plt.savefig(img_path)

# 이미지가 저장되었는지 확인
if os.path.exists(img_path): 
    print(f"✅ 그래프가 '{img_path}'로 성공적으로 저장되었습니다.") 
else: 
    print("❌ 그래프 저장에 실패했습니다.")

plt.show() # 그래프 출력

from sklearn.model_selection import train_test_split # dataSet 분리
from sklearn.linear_model import LinearRegression # 선형함수 관련 라이브러리 불러오기
import matplotlib.pyplot as plt

# 독립 변수 (X)와 종속 변수 (y)의 분리
X = df[['year']] # 년도
y = df['Net'] # 탄소량

# 데이터 분리 : 학습용(80%), 테스트용(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model = LinearRegression() # 선형모델
model.fit(X_train, y_train)

# 예측값 계산
y_pred = model.predict(X_test)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


x = df['year'].values.reshape(-1, 1)  # Reshaping to 2D array
y = df['Net']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # Use X_test for prediction

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

print("실제 값: ", list(y_test))
print("예측 값: ", list(y_pred))


# 5년 간의 Net 예측
years_to_predict = np.array([year + 5 for year in x[-5:]]).reshape(-1, 1)  # 2D 배열로 변환
future_predictions = model.predict(years_to_predict)

print("\n5년 후 예측 값: ", list(future_predictions))

# 시각화
plt.figure(figsize=(10, 6))

# 원본 데이터와 회귀선 시각화
plt.plot(x, y, color='blue', marker='.',label='Actual Net Data')
plt.plot(x, model.predict(x), color='red', label='Linear Regression Line')  # 선형 회귀 직선

# 5년 후 예측값 표시
plt.scatter(years_to_predict, future_predictions, color='green', label='Future Predictions')

plt.xlabel('Year')
plt.ylabel('Net')
plt.title('Linear Regression - Net Prediction')
plt.legend()
plt.grid(True)

# 그래프 이미지 변환 (plt.savefig(image_file_path))
img_path = 'ESG_graph_img.png'
plt.savefig(img_path)



# 이미지가 저장되었는지 확인
if os.path.exists(img_path): 
    print(f"✅ 그래프가 '{img_path}'로 성공적으로 저장되었습니다.") 
else: 
    print("❌ 그래프 저장에 실패했습니다.")

plt.show() # 그래프 출력