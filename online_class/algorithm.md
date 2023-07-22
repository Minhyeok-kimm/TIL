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
    # 모든 원소를 차례대로 힙에 삽입
    for value in iterable:
        heapq.heappush(h, value) # 내림차순의 경우 -value
    # 힙에 삽입된 모든 원소를 차례대로 꺼내어 담기
    for i in range(len(h)):
        result.append(heapq.heappop(h)) # 내림차순의 경우 -heapq.heappop(h)
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
    
    - ($N^2$ + -2) / 2로 표현이 되는데, 빅오 표기법에 따라 O($N^2$)이다.

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

    - O($N^2$). 반복문이 두 번 중첩되어 사용되기 때문이다.
    
    - 현재 리스트의 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작한다. 즉, 최선의 경우 O(N)의 시간 복잡도를 가진다.

### 3. 퀵 정렬

- 기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법

- 일반적인 상황에서 가장 많이 사용되는 정렬 알고리즘 중 하나

- 병합 정렬과 더불어 대부분의 프로그래밍 언어의 정렬 라이브러리의 근간이 되는 알고리즘(Python, C, Java)

- 가장 기본적인 퀵 정렬은 첫 번째 데이터를 기준 데이터(Pivot)로 설정

- 퀵 정렬이 빠른 이유: 이상적인 경우 분할이 절반씩 일어난다면 전체 연산 횟수로 O(NlogN)을 기대할 수 있다.

- 시간 복잡도

    - 퀵 정렬은 평균의 경우 O(NlogN)의 시간 복잡도를 가진다.
    
    - 최악의 경우 O($N^2$)의 시간 복잡도를 가진다(분할이 편향되는 경우가 해당이 된다).

- 구현 소스코드
```python
array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array, start, end):
    if start >= end: # 원소가 1개인 경우 종료
        return
    pivot = start # 첫 번째 원소에 해당
    left = start + 1
    right = end
    while(left <= right):
        # 피벗보다 큰 데이터를 찾을 때까지 반복
        while(left <= end and array[left] <= array[pivot]):
            left += 1
        # 피벗보다 작은 데이터를 찾을 때까지 반복
        while(right > start and array[right] >= array[pivot]):
            right -= 1
        if (left > right): # 엇갈렸다면 작은 데이터와 피벗을 교체
            array[right], array[pivot] = array[pivot], array[right]
        else: # 엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
            array[left], array[right] = array[right], array[left]
    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
    quick_sort(array, start, right-1)
    quick_sort(array, rkght + 1, end)

quick_sort(array, 0, len(array)-1)
print(array)
# 결과
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 파이썬의 장점을 살린 방식

def quick_sort(array):
    # 리스트가 하나 이하의 원소만을 담고 있다면 종료
    if len(array) <- 1:
        return array
    pivot = array[0]
    tail = array[1:]
    
    left_side =[x for x in tail if x <= pivot] # 분할된 왼쪽 부분
    right_side = [x for x in tail if x > pivot] # 분할된 오른쪽 부분
    
    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행하고 전체 리스트를 반환
    return quick_sort(left_side) + [pivot] + quick_sort(right_side)
```

### 4. 계수 정렬

- 특정한 조건이 부합할 때만 사용할 수 있지만 매우 빠르게 동작하는 정렬 알고리즘

    - 데이터의 크기 범위가 제한되어 정수 형태로 표현할 수 있을 때 사용이 가능하다.

- 데이터의 개수가 N, 데이터(양수) 중 최댓값이 K일 때 최악의 경우 수행시간 O(N + K)를 보장한다.

- 각 계수가 몇 번 등장했는지 conut하는 방식으로 공간 복잡도가 높다.

- 소스 코드
```python
# 모든 원소의 값이 0보다 크거나 같다고 가정
array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
# 모든 범위를 포함하는 리스트 선언(모든 값은 0으로 초기화)
count = [0] * (max(array) + 1)

for i in range(len(array)):
    count[array[i]] += 1 # 각 데이터에 해당하는 인덱스의 값 증가

for i in range(len(count)): # 리스트에 기록된 정렬 정보 확인
    for j in range(count[i]):
        print(i, end=' ') # 띄어쓰기를 구분으로 등장한 횟수만큼 인덱스 출력
```

- 복잡도 분석

    - 시간 복잡도와 공간 복잡도는 모두 O(N + K)
    
    - 계수 정렬은 때에 따라 심각한 비효율성을 초래할 수 있다(데이터가 0과 999,999 단 2개만 존재하는 경우).
    
    - 계수 정렬은 동일한 값을 가지는 데이터가 여러 개 등장할 때 효과적으로 사용할 수 있다(성적의 경우 100점을 맞은 학생이 여러 명일 수 있기 때문에 계수 정렬이 효과적이다).

###  5. 정렬 알고리즘 비교

- 대부분의 프로그래밍 언어에서 지원하는 표준 정렬 라이브러리는 최악의 경우에도 O(NlogN)을 보장하도록 설계되어 있다.

| 정렬 알고리즘 | 평균 시간 복잡도 | 공간 복잡도 | 특징 |
| ----- | ----- | ----- | ----- |
| 선택 정렬 | O($N^2$) | O(N) | 아이디어가 매우 간단하다. |
| 삽입 정렬 | O($N^2$) | O(N) | 데이터가 거의 정렬되어 있을 때는 가장 빠르다. |
| 퀵 정렬 | O(NlogN) | O(N) | 대부분의 경우에 가장 적합하며, 충분히 빠르다. |
| 계수 정렬 | O(N + K) | O(N + K) | 데이터의 크기가 한정되어 있는 경우에만 사용이 가능하지만 매우 빠르게 동작한다. |

### 6. 정렬 문제 예제

- 문제: 현재 두 개의 배열 A와 B를 가지고 있다. 두 배열은 N개의 원소로 구성되어 있으며, 배열의 원소는 모두 자연수이다. 최대 K번의 바꿔치기 연산을 수행할 수 있는데, 바꿔치기 연산이란 배열 A에 있는 원소 하나와 배열 B에 있는 원소 하나를 골라서 두 원소를 서로 바꾸는 것을 말한다. 최종 목표는 배열 A의 모든 원소의 합이 최대가 되도록 하는 것이다. N, K, 그리고 배열 A와 B의 정보가 주어졌을 때, 최대 K번의 바꿔치기 연산을 수행하여 만들 수 있는 배열 A의 모든 원소의 합의 최댓값을 출력하는 프로그램을 작성하세요.

- 해결 아이디어

    - 배열 A는 오름차순, 배열 B는 내림차순 정렬
    
    - 이후 두 배열의 원소를 첫 번째 인덱스부터 차례로 확인하며 A의 원소가 B의 원소보다 작을 때에만 교체를 수행

- 해답 코드
```python
n, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a.sort()
b.sort(reverse=True)

for i in range(k):
    if a[i] < b[i]:
        a[i], b[i] = b[i], a[i]
    else:
        break

print(sum(a))
```

## 그래프 탐색

### 1. DFS(Depth-First Search)

- DFS는 깊이 우선 탐색이라고도 부르며 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘

- 스택 자료구조(혹은 재귀함수)를 이용하며, 구체적인 동작 과정은 다음과 같다.

    1. 탐색 시작 노드를 스택에 삽입하고 방문 처리를 한다.
    
    2. 스택의 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있으면 그 노드를 스택에 넣고 방문 처리한다. 방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 꺼낸다.
    
    3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다.

- 소스코드 예제
```python
# DFS 메서드 정의
def dfs(graph, v, visited):
    # 현재 노드 방문 처리
    visited[v] = True
    print(v, end=' ')
    # 현재 노드와 연결도니 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

# 각 노드가 연결된 정보를 표현(2차원 리스트)
graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가 방문된 정보를 표현(1차원 리스트)
visited = [False] * 9

# 정의된 DFS 함수 호출
dfs(graph, 1, visited)

# 실행 결과
1 2 7 6 8 3 4 5
```

### 2. BFS(Breadth-First Search)

- 너비 우선 탐색이라고도 부르며, 그래프에서 가까운 노드부터 우선적으로 탐색하는 알고리즘

- 큐 자료구조를 이용하며, 구체적인 동작 과정은 다음과 같다.

    1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다.
    
    2. 큐에서 노드를 꺼낸 뒤에 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리한다.
    
    3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다.

- 소스코드 예제
```python
from collection import deque

# BFS 메서드 정의
def bfs(graph, v, visited):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([start])
    # 현재 노드를 방문 처리
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력하기
        v = queue.popleft()
        print(v, end=' ')
        # 아직 방문하지 않은 인접한 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

# 각 노드가 연결된 정보를 표현(2차원 리스트)
graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가 방문된 정보를 표현(1차원 리스트)
visited = [False] * 9

# 정의된 BFS 함수 호출
bfs(graph, 1, visited)

# 실행 결과
1 2 3 8 7 4 5 6
```

### 3. 문제 예제

- 문제 1: N * M 크기의 얼음 틀이 있다. 구멍이 뚫려 있는 부분은 0, 칸막이가 존재하는 부분은 1로 표시된다. 구멍이 뚫려있는 부분까지 상, 하, 좌, 우로 붙어있는 경우 서로 연결되어 있는 것으로 간주한다. 이때 얼음 틀의 모양이 주어졌을 때 생성되는 총 아이스크림의 개수를 구하는 프로그램을 작성하세요.

- 해결 아이디어

    - DFS 혹은 BFS로 해결. 그래프 형태로 모델링

- 해답 코드
```python
# dfs로 특정 노드를 방문하고 연결된 모든 노드들도 방문
def dfs(x, y):
    # 주어진 범위를 벗어나는 경우에는 즉시 종료
    if x <= -1 or x >= n or y <= -1 or y >= m:
        return False
    # 현재 노드를 아직 방문하지 않았다면
    if graph [x][y] == 0:
        # 해당 노드 방문 처리
        graph[x][y] = 1
        # 상, 하, 좌, 우의 위치들도 모두 재귀적으로 호출
        dfs(x-1, y)
        dfs(x, y-1)
        dfs(x+1, y)
        dfs(x, y+1)
        return True
    return False

# N, M을 공백을 기준으로 구분하여 입력 받기
n, m = map(int, input().split())

# 2차원 리스트의 맵 정보 입력 받기
graph = []
for i in range(n):
    graph.append(list(map(int, input())))

# 모든 노드(위치)에 대하여 음료수 채우기
result = 0
for i in range(n):
    for j in range(m):
        # 현재 위치에서 DFS 수행
        if dfs(i, j) == True:
            result += 1

# 결과 출력
print(result)
```

- 문제 2: N * M 크기의 직사각형 미로에 갇혀있는 상태이며, 미로에는 여러 마리의 괴물이 있어 이를 피해 탈출해야 한다. 현재 위치는 (1, 1)이며 미로의 출구는 (N, M)의 위치에 존재하고 한 번에 한 칸씩 이동할 수 있다. 이때 괴물이 있는 부분은 0, 없는 부분은 1로 표시되어 있다. 미로는 반드시 탈출할 수 있는 형태로 제시될 때, 탈출하기 위하여 움직여야 하는 최소 칸의 개수를 구하라. (칸을 셀 때는 시작 칸과 마지막 칸을 모두 포함하여 계산한다.)

- 해결 아이디어

    - BFS는 시작 지점에서 가까운 노드부터 차례대로 그래프의 모든 노드를 탐색한다.
    
    - 상, 하, 좌, 우로 연결된 모든 노드로의 거리가 1로 동일하기 때문에 (1, 1) 지점부터 BFS를 수행하여 모든 노드의 최단 거리 값을 기록하면 해결할 수 있다.

- 해답 코드
```python
from collections import deque
# bfs 소스코드 구현
def bfs(x, y):
    # 큐 구현을 위해 deque 라이브러리 이용
    queue = deque()
    queue.append((x, y))
    # 큐가 빌 때까지 반복
    while queue:
        x, y = queue.popleft()
        # 현재 위치에서 4가지 방향으로의 위치 확인
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            # 미로 찾기 공간을 벗어난 경우 무시
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            # 벽인 경우 무시
            if graph[nx][ny] == 0:
                continue
            # 해당 노드를 처음 방문하는 경우에만 최단거리 기록
            if graph[nx][ny] == 1:
                graph[nx][ny] = graph[x][y] + 1
                queue.append((nx, ny))
    # 가장 오른 쪽 아래까지의 최단거리 반환
    return graph[n - 1][m - 1]

# N, M을 공백을 기준으로 구분하여 입력 받기
n, m = map(int, input().split())
# 2차원 리스트의 맵 정보 입력 받기
graph = []
for i in range(n):
    graph.append(list(map(int, input())))

# 이동할 네 가지 방향 정의(상, 하, 좌, 우)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# BFS를 수행한 결과 출력
print(dfs(0, 0))
```

## 최단 경로 알고리즘

- 최단 경로 문제

    - 최단 경로 알고리즘은 가장 짧은 경로를 찾는 알고리즘을 의미한다.
    
    - 다양한 문제 상황
    
        - 한 지점에서 다른 한 지점까지의 최단 경로
        
        - 한 지점에서 다른 모든 지점까지의 최단 경로
        
        - 모든 지점에서 다른 모든 지점까지의 최단 경로
    
    - 각 지점은 그래프에서 노드로 표현
    
    - 지점 간 연결된 도로는 그래프에서 간선으로 표현

### 다익스트라 최단 경로 알고리즘

- 특정한 노드에서 출발하여 다른 모든 노드로 가는 최단 경로를 계산한다.
    
- 다익스트라 최단 경로 알고리즘은 음의 간선이 없을 때 정상적으로 동작한다.
    
    - 현실 세계의 도로(간선)은 음의 간선으로 표현되지 않는다.
        
- 다익스트라 최단 경로 알고리즘은 그리디 알고리즘으로 분류된다.
    
    - 매 상황에서 가장 비용이 적은 노드를 선택해 임의의 과정을 반복한다.
    
- 동작 과정
    
    1. 출발 노드를 설정
        
    2. 최단 거리 테이블을 초기화
        
    3. 방문하지 않은 노드 중 최단 거리가 가장 짧은 노드를 선택
        
    4. 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리 테이블을 갱신한다.
        
    5. 위 과정에서 3번과 4번을 반복한다.
    
- 알고리즘 동작 과정에서 최단 거리 테이블은 각 노드에 대한 현재까지의 최단 거리 정보를 가지고 있다. 처리 과정에서 더 짧은 경로를 찾게 되면 해당 경로가 제일 짧은 경로인 것으로 갱신한다.
    
- 특징
    
    - 그리디 알고리즘: 매 상황에서 방문하지 않은 가장 비용이 적은 노드를 선택해 임의의 과정을 반복한다.
        
    - 단계를 거치며 한 번 처리된 노드의 최단 거리는 고정되어 더이상 바뀌지 않는다. -> 한 단계당 하나의 노드에 대한 최단 거리를 확실히 찾는 것으로 이해할 수 있다.
        
    - 다익스트라 알고리즘을 수행한 뒤 테이블에 각 노드까지 최단 거리 정보가 저장된다. -> 완벽한 형태를 원하는 경우, 소스코드에 추가적인 기능을 더 넣어야 한다.
    
- 간단한 구현 방법: 단계마다 방문하지 않은 노드 중 최단 거리가 가장 짧은 노드를 선택하기 위해 매 단계마다 1차원 테이블의 모든 원소를 확인(순차 탐색)한다.
    
```python
import sys
input = sys.stdin.readline
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수를 입력받기
n, m = map(int, input().split())
# 시작 노드 번호를 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n+1)]p
# 방문한 적이 있는지 체크하는 목적의 리스트를 만들기
visited = [False] * (n+1)
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
    a, b, c = map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b, c))
    
# 방문하지 않은 노드 중, 가장 최단 거리가 짧은 노드의 번호를 반환
def get_smallest_node():
    min_value = INF
    index = 0 # 가장 최단 거리가 짧은 노드(인덱스)
    for i in range(1, n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i
    return index

def dijksstra(start):
    # 시작 노드에 대해서 초기화
    distance[start] = 0
    visited[start] = True
    for j in graph[start]:
        distance[j[0]] = j[1]
    # 시작 노드를 제외한 전체 n-1개의 노드에 대해 반복
    for i in range(n-1):
        # 현재 최단 거리가 가장 짧은 노드를 꺼내 방문 처리
        now = get_smallest_node()
        visited[now] = True
        # 현재 노드와 연결된 다른 노드를 확인
        for j in graph[now]:
            cost = distance[now] + j[1]
            # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[j[0]]:
                distance[j[0]] = cost
    
# 다익스트라 알고리즘을 수행
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리를 출력
for i in range(1, n+1):
    # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
    if distance[i] == INF:
        print('INFINITY')
    #도달할 수 있는 경우 거리를 출력
    else:
        pritn(distance[i])
```
    
- 간단한 구현 방법 성능 분석

    - 총 O(V)번에 걸쳐 최단 거리가 가장 짧은 노드를 매번 선형 탐색
        
    - 전체 시간 복잡도는 O($V^2$)이다.
        
    - 일반적으로 코딩 테스트의 최단 경로 문제에서 전체 노드가 5000개 이하라면 이 코드로 문제를 해결 가능 => 만 개가 넘어가는 경우 우선순위 큐를 이용하여 시간 복잡도를 줄여서 사용

- 개선된 구현 방법

    - 단계마다 방문하지 않은 노드 중 최단 거리가 가장 짧은 노드를 선택하기 위해 힙(Heap) 자료구조를 이용
    
    - 다익스트라 알고리즘이 동작하는 기본 원리는 동일
    
        - 최단 거리가 가장 짧은 노드를 선택해야 하므로 최소 힙을 사용한다.
        
- 개선된 구현 방법 코드

```python
import heapq
import sys
input = sys.stdin.readline
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수를 입력받기
n, m = map(int, input().split())
# 시작 노드 번호를 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n+1)]
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
    a, b, c = map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    graph[a].append((b, c))

def dijkstra(start):
    q = []
    # 시작 노드로 가기 위한 최단 경로는 0으로 설정해 큐에 삽입
    heapq.heappush(q, (0, start))
    distance[start] = 0
    while q: # 큐가 비어있지 않다면
        dist, now = heapq.heappop(q)
        # 현재 노드가 이미 처리된 적 있는 노드라면 무시
        if distance[now] < dist:
            continue
        # 현재 노드와 연결된 다른 인접한 노드들을 확인
        for i in graph[now]:
            cost = dist + i[1]
            # 현재 노드를 거쳐 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

# 다익스트라 알고리즘 수행
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리 출력
for i in range(1, n+1):
    # 도달할 수 없는 경우, 무한 출력
    if distance[i] == INF:
        print('INFINITY')
    # 도달할 수 있는 경우 거리 출력
    else:
        print(distance[i])
```

- 개선된 구현 방법 성능 분석

    - 힙 자료구조를 이용하는 다익스트라 알고리즘의 시간 복잡도는 O(ElogV)
    
    - 노드를 하나씩 꺼내 검사하는 반복문(while)은 노드의 개수 V이상의 횟수로는 처리되지 않는다.
    
        - 결과적으로 현재 우선순위 큐에서 꺼낸 노드와 연결된 다른 노드들을 확인하는 총횟수는 최대 간선의 개수(E)만큼 연산이 수행될 수 있다.
    
    - 직관적으로 전체 과정은 E개의 원소를 우선순위 큐에 넣었다가 모두 빼내는 연산과 매우 유사
    
        - 시간 복잡도를 O(ElogE)로 판단할 수 있다.
        
        - 중복 간선을 포함하지 않는 경우에 이를 O(ElogV)로 정리할 수 있다.
        
            - O(ElogE) -> O($ElogV^2$) -> O(2ElogV) -> O(ElogV)

### 플로이드 워셜 알고리즘

- 모든 노드에서 다른 모든 노드까지의 최단 경로를 모두 계산한다.

- 플로이드 워셜(Floyd-Warshall) 알고리즘은 다익스트라 알고리즘과 마찬가지로 단계별로 거쳐 가는 노드를 기준으로 알고리즘을 수행

    - 다만 매 단계마다 방문하지 않은 노드 중 최단 거리를 갖는 노드를 찾는 과정이 필요하지 않다.

- 2차원 테이블에 최단 거리 정보를 저장한다.

- 다이나믹 프로그래밍 유형에 속한다.

- 각 단계마다 특정한 노드 k를 거쳐 가는 경우를 확인한다.

    - a에서 b로 가는 최단 거리보다 a에서 k를 거쳐 b로 가는 거리가 더 짧은지 검사한다.

- 점화식: $D_{ab} = min(D_{ab}, D_{ak} + D_{kb})$

- 구현 코드
```python
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수 및 간선의 개수를 입력받기
n = int(input())
m = int(input())
# 2차원 리스트(그래프 표현)를 만들고, 무한으로 초기화
graph = [[INF] * (n+1) for _ in range(n+1)]

# 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
for a in range(1, n_1):
    for b in range(1, n+1):
        if a == b:
            graph[a][b] = 0

# 각 간선에 대한 정보를 입력 받아, 그 값으로 초기화
for _ in range(m):
    # A에서 B로 가는 비용은 C라고 설정
    a, b, c = map(int, input().split())
    graph[a][b] = c

# 점화식에 따라 플로이드 워셜 알고리즘을 수행
for k in range(1, n + 1):
    for a in range(1, n+1):
        for b in range(1, n+1):
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# 수행된 결과를 출력
for a in range(1, n+1):
    for b in range(1, n+1):
        # 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
        if graph[a][b] == INF:
            print('INFINITY', end=' ')
        # 도달할 수 있는 경우 거리를 출력
        else:
            print(graph[a][b], end=' ')
    print()
```

- 성능 분석

    - 노드의 개수가 N개일 때 알고리즘상 N번의 단계를 수행
    
        - 각 단계마다 $O(N^2)$의 연산을 통해 현재 노드를 거쳐 가는 모든 경로를 고려
    
    - 따라서 플로이드 워셜 알고리즘의 총 시간 복잡도는 $O(N^3)$이다.

### 벨만 포드(Bellma-Ford) 최단 경로 알고리즘

- 음수 간선에 관한 최단 경로 문제의 분류

    1. 모든 간선이 양수인 경우
    
    2. 음수 간선이 있는 경우
    
        - 음수 간선 순환은 없는 경우
        
        - 음수 간선 순환이 있는 경우

- 벨만 포드 최단 경로 알고리즘은 음의 간선이 포함된 상황에서도 사용할 수 있다.

    - 음수 간선의 순환을 감지할 수 있으며, 기본 시간 복잡도는 O(VE)로 다익스트라 알고리즘에 비해 느리다.

- 동작 원리: 다익스트라 알고리즘과 유사

    1. 출발 노드를 설정한다.
    
    2. 최단 거리 테이블을 초기화한다.
    
    3. 다음의 과정을 N-1번 반복한다.
    
        1. 전체 간선 E개를 하나씩 확인한다.
        
        2. 각 간선을 거쳐 다른 노드로 가는 비용을 계산하여 최단거리 테이블을 갱신한다.
    
    - 만약 음수 간선 순환이 발생하는지 체크하고 싶다면 3번의 과정을 한 번 더 수행한다. 이때 최단 거리 테이블이 갱신된다면 음수 간선 순환이 존재한다.

- 다익스트라 알고리즘과의 비교

    - 다익스트라 알고리즘
    
        - 매번 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택한다.
        
        - 음수 간선이 없다면 최적의 해를 찾을 수 있다.
    
    - 벨만 포드 알고리즘
    
        - 매번 모든 간선을 전부 확인한다. 따라서 다익스트라 알고리즘에서의 최적의 해를 항상 포함한다.
        
        - 다익스트라 알고리즘에 비해 시간이 오래 걸리지만 음수 간선 순환을 탐지할 수 있다.

- 구현 코드
```python
import sys
input = sys.stdin.readline
INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정

def bf(start):
    # 시작 노드에 대해 초기화
    dist[start] = 0
    # 전체 n번의 라운드(round)를 반복
    for i in range(n):
        # 매 반복마다 모든 간선을 확인
        for j in range(m):
            cur = edges[j][0]
            next_node = edges[j][1]
            cost = edges[j][2]
            # 현재 간선을 거쳐 다른 노드로 이동하는 거리가 더 짧은 경우
            if dist[cur] != INF and dist[next_node] > dist[cur] + cost:
                dist[next_node] = dist[cur] + cost
                # n번째 라운드에서도 값이 갱신된다면 음수 순환이 존재
                if i == n - 1:
                    return True
    return False

# 노드의 개수, 간선의 개수를 입력받기
n, m = map(int, input().split())
# 모든 간선에 대한 정보를 담는 리스트 만들기
edges = []
# 최단 거리 테이블을 모두 무한으로 초기화
dist = [INF] * (n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
    a, b, c = map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미
    edges.append((a, b, c))

# 벨만 포드 알고리즘을 수행
negative_cycle = bf(1) # 1번 노드가 시작 노드

if negative_cycle:
    print('-1')
else:
    # 1번 노드를 제외한 다른 모든 노드로 가기 위한 최던 거리 출력
    for i in range(2, n+1):
        # 도달할 수 없는 경우 -1을 출력
        if dist[i] == INF:
            print('-1')
        # 도달할 수 있는 경우 거리를 출력
        else:
            print(dist[i])
```

## 서로소 집합 알고리즘

- 서로소 집합(Disjoint Sets): 공통 원소가 없는 두 집합을 의미한다.

- 서로소 집합 자료구조

    - 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조
    
    - 서로소 집합 자료구조는 두 종류의 연산을 지원한다.
    
        - 합집합(Union): 두 개의 원소가 포함된 집합을 하나의 집합으로 합치는 연산
        
        - 찾기(Find): 특정한 원소가 속한 집합이 어떤 집합인지 알려주는 연산
    
    - 서로소 집합 자로구조는 합치기 찾기(Union Find) 자료구조라고 불리기도 한다.

    - 여러 개의 합치기 연산이 주어졌을 때 서로소 집합 자료구조의 동작 과정
    
        1. 합집합(Union) 연산을 확인하여 서로 연결된 두 노드 A, B를 확인
        
            - A와 B의 루트노드 A', B'를 각각 찾는다.
            
            - A'를 B'의 부모 노드로 설정한다.
            
        2. 모든 합집합(Union) 연산을 처리할 때까지 1번의 과정을 반복한다.

    - 서로소 집합 자료구조에서는 연결성을 통해 손쉽게 집합의 형태를 확인할 수 있다.
    
        - 기본적인 형태의 서로소 집합 자료구조에서는 루트 노드에 즉시 접근할 수 없다. <br>-> 루트 노드를 찾으려면 부모 테이블을 계속해서 확인하며 거슬러 올라가야 한다.

- 구현 방법
```python
# 특정 원소가 속한 집합 찾기
def find_parent(parent, x):
    # 루트 노드를 찾을 때까지 재귀 호출
    if parent[x] != x:
        return find_parent(parent, parent[x])
    return x

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

# 노드의 개수와 간선(Union 연산)의 개수 입력 받기
v, e = map(int, input().split())
parent = [0] * (v+1) # 부모 테이블 초기화

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v+1):
    parent[i] = [i]

# Union 연산을 각각 수행
for i in range(e):
    a, b = map(int, input().split())
    union_parent(parent, a, b)

# 각 원소가 속한 집합 출력하기
print('각 원소가 속한 집합: ', end='')
for i in range(1, v+1):
    print(find_parent(parent, i), end=' ')

print()

# 부모 테이블 내용 출력하기
print('부모 테이블: ', end='')
for i in range(1, v+1):
    print(parent[i], end=' ')
```

- 기본적인 구현 방법의 문제점

    - 합집합(Union) 연산이 편향되게 이루어지는 경우 찾기(Find) 함수가 비효율적으로 동작한다.
    
    - 최악의 경우 찾기(Find) 함수가 모든 노드를 다 확인하게 되어 시간 복잡도가 O(V)이다.

- 최적화 방법: 경로 압축(Path Compression)

    - 찾기(Find) 함수를 최적화하기 위한 방법.
    
    - 찾기(Find) 함수를 재귀적으로 호출한 뒤에 부모 테이블 값을 바로 갱신
    
    - 위 코드에서 find_parent 함수 지정만 아래와 같이 변경하면 된다.
    
    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때가지 재귀적으로 호출
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    ```
    
    - 해당 기법을 적용하면 각 노드에 대해 찾기(Find) 함수를 호출한 이후에 해당 노드의 루트 노드가 바로 부모 노드가 된다 -> 기본적인 방법에 비해 시간 복잡도가 개선된다.

- 서로소 집합을 이용한 사이클 판별

    - 무방향 그래프 내에서의 사이클을 판별할 때 사용할 수 있다.
    <br>(방향 그래프에서의 사이클 여부는 DFS를 이용해 판별 가능)
    
    - 사이클 판별 알고리즘
    
        1. 각 간선을 하나씩 확인하며 두 노드의 루트 노드를 확인한다.
        
            - 루트 노드가 서로 다르다면 두 노드에 대해 합집합(Union) 연산을 수행한다.
            
            - 루트 노드가 서로 같다면 사이클(Cycle)이 발생한 것이다.
        
        2. 그래프에 포함되어 있는 모든 간선에 대해 1번 과정을 반복한다.
    
    - 구현 코드
    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드가 아니라면, 루트 노드를 찾을 때가지 재귀적으로 호출
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parent, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
    
    # 노드의 개수와 간선(Union 연산)의 개수 입력 받기
    v, e = map(int, input().split())
    parent = [0] * (v+1) # 부모 테이블 초기화
    
    # 부모 테이블상에서 부모를 자기 자신으로 초기화
    for i in range(1, v+1):
        parent[i] = i
    
    cycle = False # 사이클 발생 여부
    
    for i in range(e):
        a, b = map(int, input().split())
        # 사이클이 발생한 경우 종료
        if find_parent(parent, a) == find_parent(parent, b):
            cycle = True
            break
        # 사이클이 발생하지 않았다면 합집합(Union) 연산 수행
        else:
            union_parent(parent, a, b)
    
    if cycle:
        print('사이클이 발생했습니다.')
    else:
        print('사이클이 발생하지 않았습니다.')
    ```
    
    ## 크루스칼 알고리즘(최소 신장 트리)
    
    - 신장 트리: 그래프에서 모든 노드를 포함하면서 사이클이 존재하지 않는 부분 그래프
    
    - 크루스칼 알고리즘
    
        - 대표적인 최소 신장 트리 알고리즘
        
        - 그리디 알고리즘으로 분류
        
        - 구체적인 동작 과정
        
            1. 간선 데이터를 비용에 따라 오름차순으로 정렬
            
            2. 간선을 하나씩 확인하며 현재의 간선이 사이클을 발생시키는지 확인
            
                - 사이클이 발생하지 않는 경우 최소 신장 트리에 포함
                
                - 사이클이 발생하는 경우 최소 신장 트리에 포함시키지 않는다.
            
            3. 모든 간선에 대해 2번의 과정을 반복

    - 구현 코드
    ```python
    # 특정 원소가 속한 집합을 찾기
    def find_parent(parent, x):
        # 루트 노드를 찾을 때까지 재귀 호출
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    
    # 두 원소가 속한 집합을 합치기
    def union_parent(parent, a, b):
        a = find_parent(parent, a)
        b = find_parent(parnet, b)
        if a < b:
            parent[b] = a
        else:
            parent[a] = b
    
    # 노드의 개수와 간선(Unoin 연산)의 개수 입력 받기
    v, e = map(int, input().split())
    parent = [0] * (v + 1) # 부모 테이블 초기화
    
    # 모든 간선을 담을 리스트와 최종 비용을 담을 변수
    edges = []
    result = 0
    
    # 부모 테이블상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v+1):
        parent[i] = i
    
    # 모든 간선에 대한 정보를 입력 받기
    for _ in range(e):
        a, b, cost = map(int, input().split())
        # 비용순으로 정렬하기 위해 튜플의 첫 번째 원소를 비용으로 설정
        edges.appent((cost, a, b))
    
    # 간선을 비용순으로 정렬
    edges.sort()
    
    # 간선을 하나씩 확인
    for edge in edges:
        cost, a, b = edge
        # 사이클이 발생하지 않는 경우에만 집합에 포함
        if find_parent(parent, a) != find_parent(parent, b):
            union_parent(parent, a, b)
            result += cost
    
    print(result)
    ```

    - 성능 분석
    
        - 간선의 개수가 E개일 때 $O(ElogE)$의 시간 복잡도를 가진다.
        
        - 가장 많은 시간을 요구하는 곳은 간선 정렬을 수행하는 부분
        <br>-> 표준 라이브러리를 이용해 E개의 데이터를 정렬하기 위한 시간 복잡도는 $O(ElogE)$이다.

## 최소 공통 조상(Lowest Common Ancestor: LCA)

- 최소 공통 조상 문제는 두 노드의 공통된 조상 중 가장 가까운 조상을 찾는 문제이다.

- 과정

    1. 모든 노드에 대한 깊이(depth)를 계산
    
    2. 최소 공통 조상을 찾을 두 노드를 확인
    
        - 먼저 두 노드의 깊이(depth)가 동일하도록 거슬러 올라간다.
        
        - 이후 부모가 같아질 때까지 반복적으로 두 노드의 부모 방향으로 거슬러 올라간다.
    
    3. 모든 LCA(a, b) 연산에 대해 2번의 과정을 반복한다.

- 구현 방법
```python
import sys
sys.setrecursionlimit(int(1e5)) # 런타임 오류 피하기
n = int(input())

parent = [0] * (n+1) # 부모 노드 정보
d = [0] * (n+1) # 각 노드까지의 깊이
c = [0] * (n+1) # 각 노드의 깊이가 계산되었는지 여부
graph = [[] for _ in range(n+1)] # 그래프 정보

for _ in range(n-1):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# 루트 노드부터 시작해 깊이(depth)를 구하는 함수
def dfs(x, depth):
    c[x] = True
    d[x] = depth
    for y oin graph[x]:
        if c[y]: # 이미 깊이를 구했다면 넘기기
            continue
        parent[y] = x
        dfs(y, depth+1)

# A와 B의 최소 공통 조상을 찾는 함수
def lca(a, b):
    # 먼저 깊이(depth)가 동일하도록 설정
    while d[a] != d[b]:
        if d[a] > d[b]:
            a = parent[a]
        else:
            b = parent[b]
    # 노드가 같아지도록 계산
    while a != b:
        a = parent[a]
        b = parent[b]
    return a

dfs(1, 0) # 루트 노드는 1번 노드

m = int(input())

for i in range(m):
    a, b = map(int, input().split())
    print(lca(a, b))
```

- 성능 분석

    - 매 쿼리마다 부모 방향으로 거슬러 올라가기 위해 최악의 경우 $O(N)$의 시간 복잡도가 요구 된다
    <br>-> 따라서 모든 쿼리를 처리할 때의 시간 복잡도는 $O(NM)$이다.

- 알고리즘 개선

    - 각 노드가 거슬러 오랄가는 속도를 빠르게 만드는 방법
    
    - 2의 제곱 형태로 거슬러 올라가도록 하면 $O(logN)$의 시간 복잡도를 보장할 수 있다. -> 메모리를 조금 더 사용하여 각 노드에 대해 $2^i$번째 부모에 대한 정보를 기록
    
    - 구현 코드
    ```python
    import sys
    input = sys.stdin.readline # 시간 초과를 피하기 위한 빠른 입력 함수
    sys.setrecursionlimit(int(1e5)) # 런타임 오류 피하기
    LOG = 21 # 2^20 = 1,000,000
    
    n = int(input())    
    parent = [[0] * LOG for _ in range(n+1)] # 부모 노드 정보
    d = [0] * (n+1) # 각 노드까지의 깊이
    c = [0] * (n+1) # 각 노드의 깊이가 계산되었는지 여부
    graph = [[] for _ in range(n+1)] # 그래프 정보

    for _ in range(n-1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)

    # 루트 노드부터 시작해 깊이(depth)를 구하는 함수
    def dfs(x, depth):
        c[x] = True
        d[x] = depth
        for y oin graph[x]:
            if c[y]: # 이미 깊이를 구했다면 넘기기
                continue
            parent[y] = x
            dfs(y, depth+1)
    
    # 전체 부모 관계를 설정하는 함수
    def set_parent():
        dfs(1, 0) # 루트 노드는 1번 노드
        for i in range(1, LOG):
            for j in range(1, n+1):
                parent[j][i] = parent[parent[j][i-1]][i-1]

    # A와 B의 최소 공통 조상을 찾는 함수
    def lca(a, b):
        # b가 더 깊도록 설정
        if [a] > d[b]:
            a, b = b, a
        # 먼저 깊이(depth)가 동일하도록 설정
        for i in range(LOG-1, -1, -1):
            if d[b] - d[a] >= (1 << i):
                b = parent[b][i]
        # 부모가 같아지도록 설정
        if a == b:
            return a;
        for i in range(LOG-1, -1, -1):
            # 조상을 향해 거슬러 올라가기
            if parent[a][i] != parent[b][i]:
                a = parent[a][i]
                b = parent[b][i]
            # 이후 부모가 찾고자 하는 조상 정보를 return
            return parent[a][0]
    
    set_parent()

    m = int(input())

    for i in range(m):
        a, b = map(int, input().split())
        print(lca(a, b))
    ```
    
    - 성능 분석
    
        - 다이나믹 프로그래밍을 이용해 시간 복잡도를 개선(세그먼트 트리를 이용하는 방법도 있음).
        
        - 매 쿼리마다 부모를 거슬러 올라가기 위해 $O(logN)$의 복잡도가 필요<br> -> 즉, 모든 쿼리를 처리할 때 시간 복잡도는 $O(MlogN)$이다.

## 위상 정렬

- 사이클이 없는 방향 그래프의 모든 노드를 방향성에 거스르지 않도록 순서대로 나열하는 것(예. 선수과목을 고려한 학습 순서 설정)

- 진입차수와 진출차수

    - 진입차수(Indegree): 특정한 노드로 들어오는 간선의 개수
    
    - 진출차수(Outdegree): 특정한 노드에서 나가는 간선의 개수

- 큐를 이용한 위상 정렬 알고리즘의 동작 과정

    1. 진입차수가 0인 모든 노드를 큐에 넣는다.
    
    2. 큐가 빌 때까지 다음의 과정을 반복한다.
    
        - 큐에서 원소를 꺼내 해당 노드에서 나가는 간선을 그래프에서 제거한다.
        
        - 새롭게 진입차수가 0이 된 노드를 큐에 넣는다.
    
    => 결과적으로 각 노드가 큐에 들어온 순서가 위상 정렬을 수행한 결과와 같다.

- 특징

    - 순환하지 않는 방향 그래프(DAG)에서만 수행할 수 있다.
    
    - 여러 가지 답이 존재할 수 있다.
    
        - 한 단게에서 큐에 새롭게 들어가는 원소가 2개 이상인 경우가 있다면 여러 가지 답이 존재
    
    - 모든 원소를 방문하기 전에 큐가 빈다면 사이클이 존재한다고 판단할 수 있다.
    
    - 스택을 활용한 DFS를 이용해 위상 정렬을 수행할 수도 있다.

- 구현 코드
```python
from collections import deque

# 노드의 개수와 간선의 개수를 입력받기
v, e = map(int, input().split())
# 모든 노드에 대한 진입차수는 0으로 초기화
indegree = [0] * (v+1)
# 각 노드에 연결된 간선 정보를 담기 위한 연결 리스트 초기화
graph [[] for i in range(v+1)]

# 방향 그래프의 모든 간선 정보를 입력 받기
for _ in range(e):
    a, b = map(int, input().split())
    graph[a].append(b) # 정점 A에서 B로 이동 가능
    # 진입 차수를 1 증가
    indegree[b] += 1

# 위상 정렬 함수
def topology_sort():
    result = [] # 알고리즘 수행 결과를 담을 리스트
    q = deque() # 큐 기능을 위한 deque 라이브러리 사용
    # 처음 시작할 때는 진입차수가 0인 노드를 큐에 삽입
    for i in range(1, v+1):
        if indegree[i] == 0:
            q.append(i)
    # 큐가 빌 때까지 반복
    while q:
        # 큐에서 원소 꺼내기
        now = q.popleft()
        result.append(now)
        # 해당 원소와 연결된 노드들의 진입차수에서 1 빼기
        for i in graph[now]:
            indegree[i] -= 1
            # 새롭게 진입차수가 0이 되는 노드를 큐에 삽입
            if indegree[i] == 0:
                q.append(i)
    # 위상 정렬을 수행한 결과 출력
    for i in result:
        print(i, end=' ')

topology_sort()
```

- 성능 분석

    - 위상 정렬을 위해 차례대로 모든 노드를 확인하며 각 노드에서 나가는 간선을 차례대로 제거해야 하므로 시간 복잡도는 $O(V + E)$이다.

## 재귀 함수(Recursive Function)

- 자기 자신을 다시 호출하는 함수를 의미한다.

- 종료 조건

    - 문제 풀이에 사용할 때 재귀 함수의 종료 조건을 반드시 명시해야 한다. 이를 명시하지 않으면 함수가 무한히 호출되어 오류 혹은 예상하지 못한 결과가 나타날 수 있다.

    - 종료 조건을 포함한 재귀함수 예제
    ```python
    def recursive_function(i):
        # 100번째 호출을 했을 때 종료되도록 종료 조건 명시
        if i = 100:
            return
        print(i, '번재 재귀함수에서', i+1, '번째 재귀함수를 호출합니다.')
        recursive_function(i + 1)
        print(i, '번째 재귀함수를 종료합니다.')
    
    recursive_function(1)
    ```

- 팩토리얼 구현 예제

    - $n! = 1 * 2 * 3 * ... * (n - 1) * n$
    
    - 수학적으로 0!과 1!의 값은 1이다.
    
    ```python
    # 반복적으로 구현한 n!
    def factorial_iterative(n):
        result = 1
        # 1부터 n까지의 수를 차례대로 곱하기
        for i in range(1, n+1):
            result *= i
        return result
    
    # 재귀적으로 구현한 n!
    def factorial_recursive(n):
        if n <= 1: # n이 1 이하인 경우 1을 반환
            return 1
        # n! = n * (n-1)!를 그대로 코드로 작성하기
        return n * factorial_recursive(n - 1)
    ```

- 유클리드 호제법 예제(최대 공약수 계산)

    - 두 자연수 A, B에 대해 (A > B) A를 B로 나눈 나머지를 R이라고 할 때, A와 B의 최대공약수는 B와 R의 최대 공약수와 같다.
    
    ```python
    def gcd(a, b):
        if a % b == 0:
            return b
        else:
            return gcd(b, a%b)
    
    print(gcd(192, 162))
    
    # 결과
    6
    ```

- 유의 사항

    - 재귀 함수를 잘 활용하면 복잡한 알고리즘을 간결하게 작성할 수 있다. 단, 오히려 다른 사람이 이해하기 어려운 형태의 코드가 될 수 있기 때문에 신중하게 사용해야 한다.
    
    - 모든 재귀 함수는 반복문을 이용하여 동일한 기능을 구현할 수 있다.
    
    - 재귀 함수가 반복문보다 유리한 경우도 있고 불리한 경우도 있다.
    
    - 컴퓨터가 함수를 연속적으로 호출하면 컴퓨터 메모리 내부의 스택 프레임에 쌓인다.
    <br>-> 스택을 사용해야 할 때 구현상 스택 라이브러리 대신에 재귀 함수를 이용하는 경우가 많다.

## 실전에서 유용한 표준 라이브러리

- 내장 함수: 기본 입출력 함수부터 정렬 함수까지 기본적인 함수들을 제공(필수적인 기능 포함)

    - 자주 사용되는 내장 함수: sum, min, max, eval, sorted, sorted with key

- itertools: 파이썬에서 반복되는 형태의 데이터를 처리하기 위한 유용한 기능들을 제공

    - 순열과 조합 라이브러리는 코딩 테스트에서 자주 사용
    
        - 순열: 서로 다른 n개에서 서로 다른 r개를 선택해 일렬로 나열하는 것
        
            - 순열의 수: ${n}P{r} = n * (n-1) * (n-2) * ... * (n-r+1)$
            
            - 구현 코드
            ```python
            from itertools import permutations
            
            data = ['A', 'B', 'C'] # 데이터 준비
            
            result = list(permutations(data, 3)) # 모든 순열 구하기
            print(result)
            ```
        
        - 조합: 서로 다른 n개에서 순서에 상관 없이 서로 다른 r개를 선택하는 것
        
            - 조합의 수: ${n}C{r} = \frac{{n}P{r}}{r!}$
            
            - 구현 코드
            ```python
            from itertools import combinations
            
            data = ['A', 'B', 'C'] # 데이터 준비
            
            result = list(combinations(data, 2)) # 2개를 뽑는 모든 조합 구하기
            print(result)
            ```
        
        - 중복 순열과 중복 조합 구현 코드
        ```python
        # 중복 순열
        from itertools import product
        
        data = ['A', 'B', 'C'] # 데이터 준비
        
        result = list(product(data, repeat=2)) # 2개를 뽑는 모든 순열 구하기(중복 허용)
        print(result)
        
        # 중복 조합
        from itertools import combinations_with_replacement
        
        data = ['A', 'B', 'C'] # 데이터 준비
        
        result list(combinations_with_replacement(data, 2)) # 2개를 뽑는 모든 조합 구하기(중복 허용)
        print(result)
        ```

- heapq: 힙(Heap) 자료구조를 제공한다.

    - 일반적으로 우선순위 큐 기능을 구현하기 위해 사용된다.

- bisect: 이진 탐색(Binary Search) 기능을 제공

- collections: 덱(deque), 카운터(Counter) 등의 유용한 자료구조를 포함한다.

    - Counter
    
        - 등장 횟수를 세는 기능을 제공한다.
        
        - 리스트와 같은 반복 가능한(iterable) 객체가 주어졌을 때 내부의 원소가 몇 번씩 등장했는지를 알려준다.
        
        - 구현 코드
        ```python
        from collections import Counter
        
        counter = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
        
        print(counter['blue'])
        print(counter['green'])
        print(dict(counter))
        
        # 결과
        3
        1
        {'red':2, 'blue':3, 'green':1}
        ```

- math: 필수적인 수학적 기능을 제공한다.

    - 팩토리얼, 제곱근, 최대공약수(GCD), 삼각함수 관련 함수부터 파이(pi)와 같은 상수를 포함한다.
    
    - 최대 공약수와 최소 공배수
    
        - gcd() 함수를 이용할 수 있다.
        ```python
        import math
        
        # 최소 공배수(LCM)를 구하는 함수
        def lcm(a, b):
            return a * b // math.gcd(a, b)
        
        a = 21
        b = 14
        
        print(math.gcd(21, 14)) # 최대 공약수(GCD) 계산
        print(lcm(21, 14)) # 최소 공배수(LCM 계산)
        ```

## 소수 여부를 빠르게 처리하는 알고리즘들

- 소수(Prime Number)

    - 1보다 큰 자연수 중 1과 자기 자신을 제외한 자연수로는 나누어 떨어지지 않는 자연수
    
- 기본적인 알고리즘 구현 코드
```python
# 소수 판별 함수(2 이상의 자연수에 대해)
def i _prime_number(x):
    # 2부터 (x - 1)까지의 모든 수를 확인
    for i in range(2, x):
        # x가 해당 수로 나누어 떨어진다면
        if x % i == 0:
            return False # 소수가 아님
    return True # 소수임
    
print(is_prime_number(4)) # False
print(is_prime_number(7)) # True
```

- 기본적인 알고리즘 성능: 2부터 X-1까지의 모든 자연수에 대해 연산을 수행해야 하므로 시간 복잡도는 $O(X)$이다.

- 약수의 성질을 이용하면 시간 복잡도를 줄일 수 있다.

    - 약수의 성질: 모든 약수가 가운데 약수를 기준으로 곱셈 연산에 대해 대칭을 이룬다.

- 개선된 알고리즘 구현 코드
```python
import math

# 소수 판별 함수(2 이상의 자연수에 대해)
def is_prime_number(x):
    # 2부터 x의 제곱근까지의 모든 수를 확인
    for i in range(2, int(math.sqrt(x)) + 1):
        # x가 해당 수로 나누어 떨어진다면
        if x%1 == 0:
            return False # 소수가 아님
    return True # 소수임

print(is_prime_number(4)) # False
print(is_prime_number(7)) # True
```

- 개선된 알고리즘 성능: 2부터 X의 제곱근(소수점 이하 무시)까지의 모든 자연수에 대해 연산을 수행해야 하므로 시간 복잡도는 $O(N^\frac{1}{2})$이다.

- 다수의 소수 판별: 에라토스테네스의 체 알고리즘을 사용

    - 다수의 자연수에 대해 소수 여부를 판별할 때 사용되는 대표적인 알고리즘이며, N보다 작거나 같은 모든 소수를 찾을 때 사용할 수 있다.
    
    - 동작 과정
    
        1. 2부터 N까지의 모든 자연수를 나열한다.
        
        2. 남은 수 중 아직 처리하지 않은 가장 작은 수 i를 찾는다.
        
        3. 남은 수 중 i의 배수를 모두 제거한다(i는 제거하지 않는다).
        
        4. 더 이상 반복할 수 없을 때까지 2번과 3번의 과정을 반복한다.
    
    - 구현 코드
    ```python
    import math
    
    n = 1000 # 2부터 1000까지의 모든 수에 대해 소수 판별
    # 처음엔 모든 수가 소수(True)인 것으로 초기화(0과 1은 제외)
    array = [True for i in range(n+1)]
    
    # 에라토스테네스의 체 알고리즘 수행
    # 2부터 n의 제곱근까지의 모든 수를 확인하며
    for i in range(2, int(math.sqrt(n)) + 1):
        if array[i] == True: # i가 소수인 경우(남은 수인 경우)
            # i를 제외한 i의 모든 배수를 지우기
            j = 2
            while i * j <= n:
                array[i * j] = False
                j += 1
    
    # 모든 소수 출력
    for i in range(2, n + 1):
        if array[i]:
            print(i, end=' ')
    ```

    - 성능 분석
    
        - 사실상 선형 시간에 가까울 정도로 매우 빠르다. 시간 복잡도는 $O(NloglogN)$이다.
        
        - 다수의 소수를 찾아야 하는 문제에서 효과적으로 사용될 수 있지만 각 자연수에 대한 소수 여부를 저장해야 하기 때문에 메모리가 많이 필요하다.

## 이진 탐색

- 순차 탐색과의 비교

    - 순차 탐색: 리스트 안에 있는 특정한 데이터를 찾기 위해 앞에서부터 데이터를 하나씩 확인하는 방법
    
    - 이진 탐색: 정렬되어 있는 리스트에서 탐색 범위를 절반씩 좁혀가며 데이터를 탐색하는 방법(시작점, 끝점, 중간점을 이용해 탐색 범위를 설정한다).

- 이진 탐색의 시간 복잡도

    - 단계마다 탐색 범위를 2로 나누는 것과 동일하므로 연산 횟수는 $log_{2}N$에 비례
    
    - 탐색 범위를 절반씩 줄이며, 시간 복잡도는 $O(logN)$을 보장

- 구현 코드(재귀적 구현)
```python
# 이진 탐색 소스코드 구현(재귀 함수)
def binary_search(array, target, start, end):
    if start > end:
        return None
    mid = (start + end) // 2
    # 찾은 경우 중간점 인덱스 반환
    if array[mid] == target:
        return mid
    # 중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
    elif array[mid] > target:
        return binary_search(array, target, start, mid-1)
    # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
    else:
        return binary_search(array, target, mid+1, end)

# n(원소의 개수)과 target(찾고자 하는 값)을 입력 받기
n, target = list(map(int, input().split()))
# 전체 원소 입력 받기
array = list(map(int, input().split()))

# 이진 탐색 수행 결과 출력
result = binary_search(array, target, 0, n-1)
if result == None:
    print('원소가 존재하지 않습니다.')
else:
    print(result+1)
```

- 구현 코드(반복문 구현)
```python
# 이진 탐색 소스코드 구현(반복문)
def binary_search(array, target, start, end):
    while start <= end:
        mid = (start + end) // 2
        # 찾은 경우 중간점 인덱스 반환
        if array[mid] == target:
        return mid
        # 중간점의 값보다 찾고자 하는 값이 작은 경우 왼쪽 확인
        elif array[mid] > target:
            end = mid-1
        # 중간점의 값보다 찾고자 하는 값이 큰 경우 오른쪽 확인
        else:
            start = mid+1

# n(원소의 개수)과 target(찾고자 하는 값)을 입력 받기
n, target = list(map(int, input().split()))
# 전체 원소 입력 받기
array = list(map(int, input().split()))

# 이진 탐색 수행 결과 출력
result = binary_search(array, target, 0, n-1)
if result == None:
    print('원소가 존재하지 않습니다.')
else:
    print(result+1)
```

- Python 이진 탐색 라이브러리

    - bisect_left(a, x): 정렬된 순서를 유지하면서 배열 a에 x를 삽입할 가장 왼쪽 인덱스를 반환
    
    - bisect_right(a, x): 정렬된 순서를 유지하면서 배열 a에 x를 삽입할 가장 오른쪽 인덱스를 반환
    
    - 해당 함수를 이용하여 값이 특정 범위에 속하는 데이터 개수 구하기 예시
    ```python
    from bisect import bisect_left, bisect_right
    
    # 값이 [left_value, right_value]인 데이터의 개수를 반환하는 함수
    def count_by_range(a, left_value, right_value):
        right_index = bisect_right(a, right_value)
        left_index = bisect_left(a, left_value)
        return right_index - left_index
    
    # 배열 선언
    a = [1, 2, 3, 3, 3, 3, 4, 4, 8, 9]
    
    # 값이 4인 데이터 개수 출력
    print(count_by_range(a, 4, 4)) # 2
    
    # 값이 [-1, 3] 범위에 있는 데이터 개수 출력
    print(count_by_range(a, -1, 3)) # 6
    ```

- 파라메트릭 서치(Parametric Search)

    - 최적화 문제를 결정 문제('예' 혹은 '아니오')로 바꾸어 해결하는 기법
    <br>예: 특정한 조건을 만족하는 가장 알맞은 값을 빠르게 찾는 최적화 문제
    
    - 일반적으로 코딩테스트에서 파라메트릭 서치 문제는 이진 탐색을 이용하여 해결할 수 있다.
    
    - 문제 예시 1
    
        - 길이가 일정하지 않은 떡볶이 떡이 있다. 한 봉지 안에 들어가는 떡의 총 길이는 절단기로 잘라서 맞춰준다.
        <br>절단기에 높이(H)를 지정하면 줄지어진 떡을 한 번에 절단하는데, 높이가 H보다 긴 떡은 H 위의 부분이 잘리고 낮은 떡은 잘리지 않는다.
        <br>예를 들어 19, 14, 10, 17cm인 떡이 나란히 있과 절단기 높이를 15cm로 지정하면 자른 뒤 떡의 높이는 15, 14, 10, 15cm가 되며, 잘린 떡의 길이는 4, 0, 0, 2cm이다. 이 잘린 떡을 손님이 가져가게 된다.
        <br>손님이 왔을 때 요청한 총 길이가 M일 대 적어도 M만큼의 떡을 얻기 위해 절단기에 설정할 수 있는 높이의 최댓값을 구하는 프로그램을 작성하라.
        
        - 문제 해결 아이디어: 적절한 높이를 찾을 때까지 이진 탐색을 수행해 높이 H를 반복해서 조정
        
        - 구현 코드
        ```python
        # 떡의 개수(N)와 요청한 떡의 길이(M)을 입력
        n, m = list(map(int, input().split(' ')))
        # 각 떡의 개별 높이 정보를 입력
        array = list(map(int, input().split()))
        
        # 이진 탐색을 위한 시작점과 끝점 설정
        start = 0
        end = max(array)
        
        # 이진 탐색 수행(반복적)
        result = 0
        while (start <= end):
            total = 0
            mid = (start + end) // 2
            for x in array:
                # 잘랐을 때의 떡의 양 계산
                if x > mid:
                    total += x - mid
            # 떡의 양이 부족한 경우 더 많이 자르기(왼쪽 부분 탐색)
            if total < m:
                end = mid - 1
            # 떡의 양이 충분한 경우 덜 자르기(오른쪽 부분 탐색)
            else:
                result = mid # 최대한 덜 잘랐을 때가 정답이므로, 여기에서 result에 기록
                start = mid + 1
        
        # 정답 출력
        print(result)
        ```

    - 정렬된 배열에서 특정 수의 개수 구하기 문제

        - 문제: N개의 원소를 포함하고 있는 수열이 오름차순으로 정렬되어 있다. 이때 이 수열에서 x가 등장하는 횟수를 계산하라. 예를 들어 수열 {1, 1, 2, 2, 2, 2, 3}이 있을 때 x = 2라면, 현재 수열에서 값이 2이 원소가 4개이므로 4를 출력한다.
        <br>단, 이 문제는 시간 복잡도 $O(logN)$으로 알고리즘을 설계하지 않으면 시간 초과 판정을 받는다.
        
        - 구현 코드
        ```python
        from bisect import bisect_left, bisect_right
    
        # 값이 [left_value, right_value]인 데이터의 개수를 반환하는 함수
        def count_by_range(a, left_value, right_value):
            right_index = bisect_right(a, right_value)
            left_index = bisect_left(a, left_value)
            return right_index - left_index
        
        n, x = map(int, input().split()) # 데이터의 개수 N, 찾고자 하는 값 x 입력받기
        array = list(map(int, input().split())) # 전체 데이터 입력받기
        
        # 값이 [x, x] 범위에 있는 데이터 개수 계산
        count = count_by_range(array, x, x)
        
        # 값이 x인 원소가 존재하지 않는다면
        if count == 0:
            print(-1)
        # 값이 x인 원소가 존재한다면
        else:
            print(count)
        ```

## 동적 계획법(다이나믹 프로그래밍)

- 메모리를 적절히 사용하여 수행 시간 효율성을 비약적으로 향상시키는 방법

- 이미 계산된 결과(작은 문제)는 별도의 메모리 영역에 저장하여 다시 계산하지 않도록 한다.

- 다이나믹 프로그래밍의 구현은 일반적으로 두 가지 방식(top-down, bottom-up)으로 구성

- 일반적인 프로그래밍 분야에서 동적(Dynamic)이란?

    - 자료구조에서 동적 할당(Dynamic Allocation)은 '프로그램이 실행되는 도중에 실행에 필요한 메모리를 할당하는 기법'을 의미한다.
    
    - 반면 다이나믹 프로그래밍에서 '다이나믹'은 별다른 의미 없이 사용된 용어이다.

- 다이나믹 프로그래밍의 조건

    1. 최적 부분 구조(Optimal Substructure): 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있다.
    
    2. 중복되는 부분 문제(Overlapping Subproblem): 동일한 작은 문제를 반복적으로 해결해야 한다.

- 메모이제이션(Memoization)

    - 다이나믹 프로그래밍을 구현하는 방법 중 하나
    
    - 한 번 계산한 결과를 메모리 공간에 메모하는 기법
    
        - 같은 문제를 다시 호출하면 메모했던 결과를 그대로 가져온다.
        
        - 값을 기록해 좋는다는 점에서 캐싱(Caching)이라고도 한다.

- top-down / bottom-up

    - top-down(Memoization) 방식은 하향식이라고도 하며, bottom-up 방식은 상향식이라고도 한다.
    
    - 다이나믹 프로그래밍의 전형적인 형태는 bottom-up 방식이다.
    
        - 결과 저장용 리스트는 DP 테이블이라고 부른다.
    
    - 엄밀히 말하면 메모이제이션은 이전에 계산된 결과를 일시적으로 기록해놓는 넓은 개념을 의미
    
        - 다이나믹 프로그래밍에 국한된 개념은 아니며, 한 번 계산 된 결과를 담아 놓기만 하고 다이나믹 프로그래밍을 위해 활용하지 않을 수도 있다.

- 대표적으로 해결할 수 있는 문제: 피보나치 수열

    - 점화식: $a_n = a_{n-1} + a_{n-2}, a_1 = 1, a_2 = 1$
    
    - 단순 재귀 소스코드
    ```python
    # 피보나치 함수(Fibonacci Function)을 재귀함수로 표현
    def fibo(x):
        if x == 1 or x == 2:
            return 1
        return fibo(x - 1) + fibo(x - 2)
    
    print(fibo(4)) # 4
    ```
    
    - 성능 분석: f(2)가 여러번 호출되어 지수 시간 복잡도($O(2^N)$)를 가지게 된다.
    
    - top-down 다이나믹 프로그래밍 소스코드
    ```python
    # 한 번 계산된 결과를 메모이제이션하기 위한 리스트 초기화
    d = [0] * 100
    
    # 피보나치 함수를 재귀함수로 구현(탑다운 다이나믹 프로그래밍)
    def fibo(x):
        # 종료 조건(1 혹은 2일때 1을 반환)
        if x == 1 or x == 2:
            return 1
        # 이미 계산한 적 있는 문제라면 그대로 반환
        if d[x] != 0:
            return d[x]
        # 아직 계산하지 않은 문제라면 점화식에 따라 피보나치 결과 반환
        d[x] = fibo(x-1) + fibo(x-2)
        return d[x]
    
    print(fibo(99)) # 218922995834555169026
    ```
    
    - bottom-up 다이나믹 프로그래밍 소스코드
    ```python
    d = [0] * 100
    
    # 첫 번째 피보나치 수와 두 번째 피보나치 수는 1
    d[1] = 1
    d[2] = 1
    n = 99
    
    # 피보나치 함수 반복문으로 구현
    for i in range(3, n+1):
        d[i] = d[i-1] + d[i-2]
    
    print(d[n]) # 218922995834555169026
    ```

- 다이나믹 프로그래밍 vs 분할 정복

    - 모두 최적 부분 구조를 가질 때 사용할 수 있다.
    
        - 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아 큰 문제를 해결할 수 있는 상황
    
    - 둘의 차이점은 부분 문제의 중복
    
        - 다이나믹 프로그래밍 문제에서는 각 부분 문제들이 서로 영향을 미치며 부분 문제가 중복된다.
        
        - 분할 정복 문제에서는 동일한 부분 문제가 반복적으로 계산되지 않는다(대표적으로 퀵 정렬).

## 그리디 알고리즘(탐욕법)

- 현재 상황에서 지금 당장 좋은 것만 고르는 방법을 의미한다.

- 일반적인 그리디 알고리즘은 문제를 풀기 위한 최소한의 아이디어를 떠올릴 수 있는 능력을 요구

- 정당성 분석이 중요(단순히 가장 좋아 보이는 것을 반복적으로 선택해도 최적의 해를 구할 수 있는지 검토)

- 문제(거스름 돈)

    - 문제 설명: 음식점 카운터에 거스름돈으로 사용할 500원, 100원, 50원, 10원짜리 동전이 무한히 존재한다고 가정한다. 손님에게 거슬러 주어야 할 돈이 N원일 때 거슬러 주어야 할 동전의 최소 개수를 구하라. 단, 거슬러줘야 할 돈 N은 항상 10의 배수이다.
    
    - 문제 해결 아이디어: 최적의 해를 빠르게 구하기 위해서는 가장 큰 화폐 단위부터 돈을 거슬러주면 된다.
    
    - 정당성 분석: 가장 큰 화폐 단위부터 돈을 거슬러 주는 것이 최적의 해를 보장하는 이유는 가지고 있는 동전 중 큰 단위가 항상 작은 단위의 배수이므로 작은 단위의 동전들을 종합해 다른 해가 나올 수 없기 때문이다.
    
    - 답안 예시
    ```python
    n = 1260
    count = 0
    
    # 큰 단위의 화폐부터 차례대로 확인하기
    array = [500, 100, 50, 10]
    
    for coin in array:
        count += n // coin
        n %= coin
    
    print(count)
    ```
    
    - 시간 복잡도 분석: 화폐의 종류가 K라고 할 때, 소스코드의 시간 복잡도는 $O(K)$이다. 즉, 거슬러줘야 하는 금액과 무관하고, 화폐의 종류 개수가 관련이 있다.

## 구현: 시뮬레이션과 완전 탐색

- 구현(Implementation): 머릿속에 있는 알고리즘을 소스코드로 바꾸는 과정

    - 흔히 알고리즘 대회에서 구현 유형의 문제란, 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제를 지칭한다.
    
    - 예시: 알고리즘은 간단한데 코드가 지나칠 만큼 길어지는 문제, 실수 연산을 다루고 특정 소수점 자리까지 출력해야 하는 문제, 문자열을 특정한 기준에 따라 끊어 처리해야 하는 문제, 적절한 라이브러리를 찾아 사용해야 하는 문제 등
    
    - 일반적으로 알고리즘 문제에서 2차원 공간은 행렬(Matrix)의 의미로 사용
    
    - 시뮬레이션 및 완전 탐색 문제에서는 2차원 공간에서의 방향 벡터가 자주 활용된다.

# 투 포인터(Two Pointers)

- 리스트에 순차적으로 접근해야 할 때 두 개의 점의 위치를 기록하면서 처리하는 알고리즘

- 흔히 2, 3, 4, 5, 6, 7번 학생을 지목해야 할 때 간단히 '2번부터 7번까지의 학생'이라고 부르는데, 이처럼 리스트에 담긴 데이터에 순차적으로 접근해야 할 때 시작점과 끝점 2개의 점으로 접근할 데이터의 범위를 지정하는 방식