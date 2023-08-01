# 함수
import random
def findMinIndex(ary):
    minidx = 0
    for i in range(1, len(ary)):
        if ary[minidx] > ary[i]:
            minidx = i
    return minidx

# 변수
before = [random.randint(30, 190) for _ in range(8)]
after = []

# 메인
print('정렬 전 -->', before)
for i in range(len(before)):
    minnow = findMinIndex(before)
    after.append(before[minnow])
    del before[minnow]

print('정렬 후 -->', after)