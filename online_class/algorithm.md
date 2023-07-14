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