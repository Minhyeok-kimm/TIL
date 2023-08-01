# 함수
def printStar(n):
    if n > 0:
        printStar(n-1)
        print('*' * n)

# 메인
printStar(5)