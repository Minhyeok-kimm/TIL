# 함수
def pow(x, n):
    if n == 0:
        return 1
    else:
        return x * pow(x, n-1)

# 메인
print(pow(2, 3))