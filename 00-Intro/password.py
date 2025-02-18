import re

def is_valid_password(password: str) -> bool:
    # 정규 표현식 패턴: 최소 하나의 소문자, 대문자, 숫자, 특수문자 포함
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return bool(re.match(pattern, password))

while True:
    user_input = input("문장을 입력하세요 ('!quit' 입력 시 종료): ")
    if user_input == "!quit":
        print("프로그램을 종료합니다.")
        break
    elif is_valid_password(user_input):
        print("입력한 문장:", user_input)
    else:
        print("입력한 문장: 정규 표현식에 맞지 않습니다.")