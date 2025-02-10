# ----------------------------------
# 작성자: 류지혜 
# 작성목적: ESG Data를 활용한 데이터 분석 실습
# 작성일: 2025.02.10
#
# 본 파일은 KDT 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
# 변경사항 내역(날짜, 변경목적, 변경내용 순으로 기입)
#
#
# ----------------------------------

import pandas as pd
import numpy as np
import matplotlib as plt 
import seaborn as sns

# 파일 경로 설정
file_path = "ESG_CO2_2021.csv"

# 파일 경로를 현재 디렉토리로 변경
import os
os.chdir("/config/workspace/Basic/python_code")

# 데이터 로드 (UTF-8 인토딩 시도 후 실패 시 다른 인코딩 시도
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError: 
    try:
        df = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="euc-kr")

# 로드된 데이터 기본 정보 확인
print("-"*50)
print('데이터 타입 및 결측치 확인')
print("-"*50)
df.info()

print("-"*50)
print('상위 5개행 출력') # print(df.head(3)) 3개만 출력
print("-"*50)
print(df.head())

print("-"*50)
print('요약 통계량 출력 (수치형 컬럼만)')
print("-"*50)
print(df.describe())

print("-"*50)
print('데이터 행,열 갯수 출력')
print("-"*50)
print(df.shape)