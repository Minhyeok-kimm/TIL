# 함수
def isStackFull(): # 매우 중요. overflow 방지
    global size, stack, top
    if top >= size-1:
        return True
    else:
        return False

def push(data):
    global size, stack, top
    if isStackFull():
        print('스택이 가득 찼습니다.')
        return
    top += 1
    stack[top] = data

def isStackEmpty():
    global size, stack, top
    if top == -1:
        return True
    else:
        return False

def pop():
    global size, stack, top
    if isStackEmpty():
        print('스택에 데이터가 없습니다.')
        return
    data = stack[top]
    stack[top] = None
    top -= 1
    return data

def peek():
    global size, stack, top
    if isStackEmpty():
        print('스택에 데이터가 없습니다.')
        return
    return stack[top]

# 변수
size = 5
stack = [None for _ in range(size)]
top = -1

# 메인
push('커피')
push('녹차')
push('꿀물')
# push('콜라')
# push('환타')
# print('바닥:', stack)

# push('게토레이')
# print('바닥:', stack)
print('바닥:', stack)
retData = pop()
print('pop -->', retData)
print('바닥:', stack)
print('다음 예정:', peek())