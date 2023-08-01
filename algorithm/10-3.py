# 함수
import random
def selectionSort(ary):
    n = len(ary)
    for i in range(n-1):
        minidx = i
        for j in range(i+1, n):
            if ary[minidx] > ary[j]:
                minidx = j
        ary[minidx], ary[i] = ary[i], ary[minidx]
    return ary

# 변수
dataAry = [random.randint(30, 190) for _ in range(80)]

# 메인
print('정렬 전-->', dataAry)
dataAry = selectionSort(dataAry)
print('정렬 후-->', dataAry)