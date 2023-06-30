# Algorithm

## 스택

- 먼저 들어온 데이터가 나중에 나가는 형식(선입후출)의 자료구조.

- 입구와 출구가 동일한 형태로 스택을 시각화 할 수 있다. 예) 박스 쌓기

- DFS 뿐 아니라 다양한 알고리즘에서 사용

- 스택 구현 예제(Python)

```python
stack = []

# 삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제() ⇒ 5, 2, 3, 1만 남음
stack.append(5)
stack.append(2)
stack.append(3)
stack.append(7)
stack.pop()
stack.append(1)
stack.append(4)
stack.pop()

print(stack[::-1])  # 최상단 원소부터 출력. [1, 3, 2, 5]
print(stack)        # 최하단 원소부터 출력. [5, 2, 3, 1]
```

---

## 큐

- 먼저 들어온 데이터가 먼저 나가는 형식(선입선출)의 자료구조.

- 큐는 입구와 출구가 모두 뚫려 있는 터널과 같은 형태로 시각화 할 수 있다. 예) 대기열

- 스택 구현 예제(Python)

```python
from collections import deque
# 일반 리스트로 구현하는데에는 시간 복잡도가 높기 때문에 라이브러리 사용
queue = deque()

# 삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)
queue.popleft()
queue.append(1)
queue.append(4)
queue.popleft()

print(queue)    # 먼저 들어온 순서대로 출력. deque([3, 7, 1, 4])
queue.reverse()
print(queue)    # 나중에 들어온 원소부터 출력. deque([4, 1, 7, 3])
```

---

## 우선순위 큐(Priority Queue)

- 우선순위가 가장 높은 데이터를 가장 먼저 삭제하는 자료구조.

- 데이터를 우선순위에 따라 처리하고 싶을 때 사용한다.
</br>예) 물건 데이터를 자료구조에 넣었다가 가치가 높은 물건부터 꺼내서 확인해야 하는 경우

- 구현하는 방법

    1. 리스트 이용

    2. 힙(heap)을 이용하여 구현

- 데이터의 개수가 N개일 때, 구현 방식에 따라서 시간 복잡도를 비교

| 우선순위 큐 구현 방식 | 삽입시간 | 삭제시간|
| ----- | ----- | -----|
| 리스트 | O(1) | O(N) |
| 힙(Heap) | O(logN) | O(logN) |

- 단순히 N개의 데이터를 힙에 넣었다가 모두 꺼내는 작업은 정렬과 동일하다.(힙 정렬)

  - 이 경우 시간 복잡도는 O(NlogN)이다.

- 특징

  - 완전 이진 트리 자료구조의 일종
  
    - 완전 이진 트리: 루트 노드부터 시작하여 왼쪽 자식 노드, 오른쪽 자식 노드 순서대로 데이터가 차례대로 삽입되는 트리(tree)를 의미
  
  - 항상 루트 노드(root node)를 제거한다.
  
  - 최소 힙(min heap)
  
    - 루트 노드가 가장 작은 값을 가진다. -> 값이 작은 데이터가 우선적으로 제거된다.

    - Min-Heapify() 함수를 이용 : (상향식)부모 노드로 거슬러 올라가며, 부모보다 자신의 값이 더 작은 경우, 위치를 교체. O(logN)의 시간 복잡도로 힙 성질을 유지할 수 있게 한다.

  - 최대 힙(max heap)
  
    - 루트 노드가 가장 큰 값을 가진다. -> 값이 큰 데이터가 우선적으로 제거된다.

- 구현 예제(Python - 기본적으로 Min-Heapify()가 적용)

```python
import sys
import heapq
input = sys.stdin.readline

def heapsort(iterable):
    h = []
    result = []
    # 모든 원소를 차례대로 십에 삽입
    for value in iterable:
        heapq.heappush(h, value)
    # 힙에 삽입된 모든 원소를 차례대로 꺼내어 담기
    for i in range(len(h)):
        result.append(heapq.heappop(h))
    return result

n = int(input())
arr = []

for i in range(n):
    arr.append(int(input()))

res = heapsort(arr)

for i in range(n):
    print(res[i])
```

---

## 트리(Tree)

### 1. 트리

- 가계도와 같은 계층적인 구조를 표현할 때 사용

- 트리 관련 용어

    - 루트 노드(root node): 부모가 없는 최상위 노드
    
    - 단말 노드(leaf node): 자식이 없는 노드
    
    - 크기(size): 트리에 포함된 모든 노드의 개수
    
    - 깊이(depth): 루트 노드부터의 거리
    
    - 높이(height): 깊이 중 최댓값
    
    - 차수(degree): 각 노드의 (자식 방향) 간선 개수(자식 노드의 개수)
    
- 기본적으로 트리의 크기가 N일 때 전체 간선의 개수는 N-1개이다.

### 2. 이진 탐색 트리

- 이진 탐색이 동작할 수 있도록 고안된 효율적 탐색이 가능한 자료구조

- 특징: 왼쪽 자식 노드 < 부모노드 < 오른쪽 자식 노드

    - 부모 노드보다 왼쪽 자식 노드가 작다.
    
    - 부모 노드보다 오른쪽 자식 노드가 크다.

- 데이터를 조회하는 과정(이미 구성되어 있는 경우를 가정)

    1. 루트 노드부터 방문하여 탐색을 진행: 찾는 원소와 값을 비교 후 찾는 원소가 작은 경우 왼쪽, 큰 경우 오른쪽으로 이동한다.
    
    2. 이동한 뒤 현재 노드와 값을 비교: 1번 과정과 마찬가지로 찾는 원소와 값을 비교하여 결과에 따라 이동 -> 반복

### 3. 트리의 순회

- 트리 자료구조에 포함된 노드를 특정한 방법으로 한 번씩 방문하는 방법.

    - 이 때 트리의 정보를 시각적으로 확인할 수 있다.

- 대표적인 트리 순회 방법

    - 전위 순회(pre-order traverse): 루트를 먼저 방문 후 왼쪽 방향으로 방문한다.
    
    - 중위 순회(in-order traverse): 제일 왼쪽 자식을 방문한 뒤에 루트를 방문한다. 그 후 오른쪽 자식이 존재하면 해당 노드에 방문한다.
    
    - 후위 순회(post-order traverse): 제일 왼쪽 자식을 방문한 뒤 오른쪽 자식을 방문하고 루트를 방문한다.

- 구현 예제

```python
class Node:
    def __init__(self, data, left_node, right_node):
        self.data = data
        self.left_node = left_node
        self.right_node = right_node

# 전위 순회
def pre_order(node):
    print(node.data, end=' ')
    if node.left_node != None:
        pre_order(tree[node.left_node])
    if node.right_node != None:
        pre_order(tree[node.right_node])

# 중위 순회
def in_order(node):
    if node.left_node != None:
        in_order(tree[node.left_node])
    print(node.data, end=' ')
    if node.right_node != None:
        in_order(tree[node.right_node])

# 후위 순회
def post_order(node):
    if node.left_node != None:
        post_order(tree[node.left_node])
    if node.right_node != None:
        post_order(tree[node.right_node])
    print(node.data, end=' ')

n = int(input())
tree = {}

for i in range(n):
    data, left_node, right_node = input().split()
    if left_node == "None":
        left_node = None
    if right_node == "None":
        right_node = None
    tree[data] = Node(data, left_node, right_node)

pre_order(tree['A'])
print()
in_order(tree['A'])
print()
post_order(tree['A'])
```

### 4. 바이너리 인덱스 트리

- 예시 문제: 데이터 업데이트가 가능한 상황에서 구간 합 문제

- 2진법 인덱스 구조를 활용해 구간 합 문제를 효과적으로 해결할 수 있는 자료구조. 펜윅 트리라고도 한다.

- 0이 아닌 마지막 비트를 찾는 법: 특정한 숫자 K의 0이 아닌 마지막 비트를 찾기 위해 K & -K를 계산하면 된다.

- K&-K 계산 결과 예시
```python
n = 8
for i in range(n + 1):
    print(i, "의 마지막 비트:", (i & -i))

# 결과
0의 마지막 비트: 0
1의 마지막 비트: 1
2의 마지막 비트: 2
3의 마지막 비트: 1
4의 마지막 비트: 4
5의 마지막 비트: 1
6의 마지막 비트: 2
7의 마지막 비트: 1
8의 마지막 비트: 8
```

- 바이너리 인덱스 트리 구현
```python
import sys
imput = sys.stdin.readline

# 데이터의 개수(n), 변경 횟수(m), 구간 합 계산 횟수(k)
n, m, k = map(int, input().split())

# 전체 데이터의 개수는 최대 1,000,000개
arr = [0] * (n + 1)
tree = [0] * (n + 1)

# i번째 수까지의 누적 합을 계산하는 함수
def prefix_sum(i):
    result=0
    while i > 0:
        result += tree[i]
        # 0이 아닌 마지막 비트만큼 빼가면서 이동
        i -= (i & -i)
    return result

# i번째 수를 dif만큼 더하는 함수
def update(i, dif):
    while i <= n:
        tree[i] += dif
        i += (i & -i)

# start부터 end까지의 구간 합을 계산하는 함수
def interval_sum(start, end):
    return prefix_sum(end) - prefix_sum(start - 1)

for i in range(1, n + 1):
    x = int(input())
    arr[i] = x
    update(i, x)

for i in range(m + k):
    a, b, c = map(int, input().split())
    # 업데이트(update) 연산일 경우
    if a == 1:
        update(b, c - arr[b])
        arr[b] = c
    # 구간 합(interval sum) 연산인 경우
    else:
        print(interval_sum(b, c))
```

---

## 정렬 알고리즘

- 정렬(Sorting)이란 데이터를 특정한 기준에 따라 순서대로 나열하는 것

- 일반적으로 문제 상황에 따라 적절한 정렬 알고리즘이 공식처럼 사용된다.

### 1. 선택 정렬

- 처리되지 않은 데이터 중에서 가장 작은 데이터를 **선택**해 맨 앞에 있는 데이터와 바꾸는 것을 반복한다.

- 선택 정렬 소스코드
```python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(len(array)):
    min_index = i
    for j in range(i + 1, len(array)):
        if array[min_index] > array[j]:
            min_index = j
    array[i], array[min_index] = array[min_index], array[i]

print(array)
# 결과
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

- 시간 복잡도

    - 선택 정렬은 N번 만큼 가장 작은 수를 찾아서 맨 앞으로 보낸다
    
    - (N^2 + -2) / 2로 표현이 되는데, 빅오 표기법에 따라 O(N^2)이다.

### 2. 삽입 정렬

- 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입한다.

- 선택 정렬에 비해 구현 난이도가 높지만 일반적으로 더 효율적으로 동작한다.

- 삽입정렬 소스코드
```python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(array)):
    for j in range(i, 0, -1):
        if array[j] < array[j - 1]:
            array[j], array[j - 1] = array[j - 1], array[j]
        else:
            break

print(array)
# 결과
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

- 시간 복잡도

    - O(N^2). 반복문이 두 번 중첩되어 사용되기 때문이다.
    
    - 현재 리스트의 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작한다. 즉, 최선의 경우 O(N)의 시간 복잡도를 가진다.