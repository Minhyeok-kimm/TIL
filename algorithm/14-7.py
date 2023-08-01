# 함수
def gugu(dan, n):
    print('%d x %d = %d' % (dan, n, dan*n))
    if n < 9:
        gugu(dan, n+1)

# 메인
for i in range(2, 10):
    print(f'--- {i}단 ---')
    gugu(i, 1)