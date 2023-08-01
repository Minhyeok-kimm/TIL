# 함수
def isQueueFull():
    global size, queue, front, rear
    if (rear+1) % size == front:
        return True
    else:
        return False

def enQueue(data):
    global size, queue, front, rear
    if isQueueFull():
        print('큐가 꽉 찼습니다')
        return
    rear = (rear+1)%size
    queue[rear] = data

def isQueueEmpty():
    global size, queue, front, rear
    if front == rear:
        return True
    else:
        return False

def deQueue():
    global size, queue, front, rear
    if isQueueEmpty():
        print('큐가 비어있습니다.')
        return
    front = (front+1) % size
    data = queue[front]
    queue[front] = None
    return data

def peek():
    global size, queue, front, rear
    if isQueueEmpty():
        print('큐가 비어있습니다.')
        return
    return queue[(front+1)%5]

# 변수
size = 5
queue = [None for _ in range(size)]
front = rear = -1

# 메인
enQueue('화사')
enQueue('솔라')
enQueue('문별')
enQueue('휘인')
# enQueue('선미')
# print('출구<--', queue, '<--입구')

# enQueue('재남')
print('출구<--', queue, '<--입구')

# retData = deQueue()
# print('추출:', retData)
# print('다음 추출:', peek())

retData = deQueue()
print('식사 손님 :', retData)
print('출구<--',queue,'<--입구')

retData = deQueue()
print('식사 손님 :', retData)
print('출구<--',queue,'<--입구')

enQueue('재남')
enQueue('정국')
enQueue('길동')
print('출구<--',queue,'<--입구')
