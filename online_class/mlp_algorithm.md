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