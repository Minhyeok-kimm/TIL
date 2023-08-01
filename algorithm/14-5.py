# 함수
def countdown(num):
    if num == 0:
        print('발사')
    else:
        print(num)
        countdown(num-1)

# 메인
countdown(5)