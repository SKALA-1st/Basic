# 1번 문항
user_input = input("문장을 입력하세요: ")
print("입력한 문장:", user_input)

# 2번 문항
while True:
    user_input = input("문장을 입력하세요 ('!quit' 입력 시 종료): ")
    if user_input == "!quit":
        print("프로그램을 종료합니다.")
        break
    print("입력한 문장:", user_input)
