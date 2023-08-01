# 함수
import random
def binSearch(ary, data):
    pos = -1
    start = 0
    end = len(ary) - 1
    while start <= end:
        mid = (start + end) // 2
        if ary[mid] == data:
            pos = mid
            break
        elif ary[mid] < data:
            start = mid + 1
        elif ary[mid] > data:
            end = mid - 1
    return pos

# 변수
dataAry = [random.randint(30, 190) for _ in range(10)]
findData = random.choice(dataAry)
dataAry.sort()

# 메인
print('배열 -->', dataAry)
position = binSearch(dataAry, findData)
if position == -1:
    print(findData, '가 없습니다.', sep='')
else:
    print(findData, '는 ', position, ' 위치에 있습니다.', sep='')