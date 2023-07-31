# 함수
# Node 데이터형을 정의
class Node():
    def __init__(self):
        self.data = None
        self.link = None

# 변수


# 메인
# node 추가
node1 = Node()
node1.data = '다현'

node2 = Node()
node2.data = '정연'
node1.link = node2

node3 = Node()
node3.data = '쯔위'
node2.link = node3

node4 = Node()
node4.data = '사나'
node3.link = node4

node5 = Node()
node5.data = '지효'
node4.link = node5

# new_node 생성 후 link 변경
# new_node = Node()
# new_node.data = '재남'
# new_node.link = node2.link
# node2.link = new_node

# node 삭제
node2.link = node3.link
del node3

# print를 일일히 작성하지 않고 연달아서 출력
current = node1
print(current.data, end=' ')
while current.link != None:
    current = current.link
    print(current.data, end=' ')
print()


# 출력
# print(node1.data, end = ' ')
# print(node1.link.data, end = ' ')
# print(node1.link.link.data, end = ' ')
# print(node1.link.link.link.data, end = ' ')
# print(node1.link.link.link.link.data, end = ' ')
# print(node1.link.link.link.link.link.data, end = ' ')