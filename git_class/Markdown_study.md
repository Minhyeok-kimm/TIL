# Markdown

## Basic Syntax

### 1. Heading

문서의 제목이나 소제목을 표현한다.

`#` 개수에 따라 대응되는 수준이 있고, 글자 크기 조절을 위해 사용되는 것은 바람직하지 않다.

### 2. Bold

**진한 글씨**를 표현하는 방법.

진하게 만들고 싶은 글씨 양 옆에 `**`를 입력한다.

### 3. Italic

*기울인 글씨*를 표현하는 방법.

기울이고 싶은 글씨 양 옆에 `*`을 입력한다.

### 4. Blockquote

인용문을 작성할 때 사용한다. `>`를 이용해 작성한다.

> 인용문 작성예시

### 5. List

기본적으로 리스트는 Tab을 이용해서 하위 항목으로 구별할 수 있다.

    1. Ordered List(ol)
    
    순서가 존재하는 리스트
    
    `1. 2. 3.`을 이용해서 순서를 나타내어 작성한다.
    
    2. Unordered List(ul)
    
    순서가 존재하지 않는 리스트.
    
    `-`를 사용하여 작성한다.

### 6. Code

`inline code block을 만든다.`

backtick 기호 사이에 해당하는 문자를 넣는다.

추가로, Escaping backtick는 backtick를 2개 사용하면 된다.

`` `code`를 통해 코드를 작성할 수 있다. ``

### 7. Horizotal Rule

수평선을 삽입한다. 하이픈 3개를 사용한다.

---

### 8. Link

인터넷 링크 등을 하이퍼링크로 삽입하는 방법이다.

대괄호 안에 해당 링크의 이름, 그 뒤에 소괄호로 해당하는 링크를 넣는다. 로컬 파일 등으로도 지정할 수 있다.

[마크다운 가이드](https://www.markdownguide.org/)

### 9. Image

이미지를 삽입하는 방법이다.

Link를 만드는 방법과 동일하게 작성하나, 맨 앞에 `!`를 추가한다.

![캡쳐 사진](git_class/screenshot.png)

---

## Extended Syntax

### 1. Fenced Code Block

파이썬, SQL, HTML 등 다양한 코드 블럭을 만드는 방법.

backtick 3개 사이에 입력을 하며, 언어를 입력해두면 해당 언어를 표현하게 된다.

```python
print('hello')
```

``` SQL
SELECT * FROM DEPT;
```

### 2. Table

표를 작성하는 방법이다.

파이프 기호를 이용해서 만들 수 있다.

| 이름 | 나이 |
| ----- | ----- |
| 홍길동 | 24 |
| 김철수 | 32 |

### 3. Highlight

문장에서 <mark>강조</mark>하고 싶은 경우 이용한다.

`<mark> 강조할 문장 <mark>`의 방식으로 사용한다.
