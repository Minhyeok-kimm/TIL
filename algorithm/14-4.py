# 함수
def Factorial(num):
    if num == 1:
        return 1
    return num * Factorial(num-1)

# 메인
print(Factorial(10))