# 함수
class TreeNode():
    def __init__(self):
        self.left = None
        self.data = None
        self.right = None

# 변수
memory = []
root = None
nameArray = ['블랙핑크', '레드벨벳', '마마무', '에이핑크', '걸스데이', '트와이스']

# 메인
node = TreeNode()
node.data = nameArray[0]
root = node
memory.append(node) # 중요하지 않음

# 가나다 순 기준 이진 탐색 트리 생성
for name in nameArray[1:]:
    node = TreeNode()
    node.data = name
    
    current = root
    
    while True:
        if name < current.data:
            if current.left == None:
                current.left = node
                break
            else:
                current = current.left
        else:
            if current.right == None:
                current.right = node
                break
            else:
                current = current.right
    
    memory.append(node)

print('이진 탐색 트리 완료')

findName = '마마무'
current = root
while True:
    if current.data == findName:
        print(findName, '찾음')
        break
    elif findName < current.data:
        if current.left == None:
            print(findName, '없음')
            break
        current = current.left
    else:
        if current.right == None:
            print(findName, '없음')
            break
        current = current.right