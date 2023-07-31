# 함수
class Node():
    def __init__(self):
        self.data = None
        self.link = None

def printNodes(start):
    current = start
    print(current.data, end=' ')
    while current.link != None:
        current = current.link
        print(current.data, end=' ')
    print()

def insertNode(findData, insertData):
    global memory, head, current, pre
    # case1: 현재 head 안에 삽입할 때
    if head.data == findData:
        node = Node()
        node.data = insertData
        node.link = head
        head = node
        memory.append(node) # 중요하지 않음
        return
    # case2: 중간 노드 앞에 삽입할 때
    current = head
    while current.link != None:
        pre = current
        current = current.link
        if current.data == findData:
            node = Node()
            node.data = insertData
            node.link = current
            pre.link = node
            memory.append(node) # 중요하지 않음
            return
    # case3: 마지막 노드 삽입(없는 노드 앞에 삽입)
    node = Node()
    node.data = insertData
    current.link = node
    memory.append(node)
    return

def deleteNode(data):
    global memory, current, head, pre
    # case1: head 노드 삭제
    if head.data == data:
        current = head
        head = head.link
        del current
        return
    # case2: 중간 노드 삭제
    current = head
    while current.link != None:
        pre = current
        current = current.link
        if current.data == data:
            pre.link = current.link
            del current
            return
    # case3: 삭제할 노드가 없을 때
    return

def findNode(findData):
    global memory, head, current, pre
    current = head
    if current.data == findData:
        return current
    while current.link != None:
        current = current.link
        if current.data == findData:
            return current
    # 파이썬은 유연해서 설정하지 않아도 되지만, 다른 언어의 경우 return 형식을 맞춰주는게 좋다.
    return Node()

# 전역
memory = []
head, current, pre = None, None, None
dataArray = ['다현', '정연', '쯔위', '사나', '지효'] # 실제 사용 데이터 모음

# 메인
# 헤드노드 생성
node = Node()
node.data = dataArray[0]
head = node

# 메모리에 추가
memory.append(node) # 중요하지 않음

for i in dataArray[1:]:
    pre = node
    node = Node()
    node.data = i
    pre.link = node
    memory.append(node)

printNodes(head)

# insertNode('다현', '화사')
# printNodes(head)

# insertNode('사나', '솔라')
# printNodes(head)

# insertNode('재남', '문별')
# printNodes(head)

# deleteNode('다현')
# printNodes(head)

# deleteNode('쯔위')
# printNodes(head)

# deleteNode('재남')
# printNodes(head)

fNode = findNode('사나')
print(fNode.data, '뮤비가 나옵니다')