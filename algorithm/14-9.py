# 함수
import random

def arySum(ary, n):
    if n <= 0:
        return ary[0]
    return ary[n] + arySum(ary, n-1)

# 메인
array = [random.randint(0, 255) for _ in range(random.randint(10, 20))]
print('배열-->', array)
print('배열의 총 합:', arySum(array, len(array)-1))