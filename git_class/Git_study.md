# 6/8 복습 문제

## 1. 내가 작업을 완료하고 버전으로 기록하기 위해 실행하는 명령어를 작성해주세요.

git add .

git commit -m '버전 기록'

> `git add <file/directory>`
>
> `git commit -m 'message'`

## 2. 아래 표현의 의미를 작성하세요.

Untracked - 커밋이 한 번도 되지 않은 파일로, 현재 버전 관리가 되고 있지 않은 상태

> 한 번도 commit이 된 적 없는 파일

Changes not staged for commit - staging area에 아직 현재 버전이 올라가지 않은 상태

> working directory에서 변경점은 있지만 staging area에는 기록되지 않은 상태

nothing to commit, working tree clean - commit을 할 버전이 없는 상태

> 현재 달라진 상태가 없는 상태

## 3. Git은 무엇인가?

분산형 버전 관리 시스템

> 형상관리, 버전 관리 / 분산형

## 4. GitHub은 무엇인가?

Git을 온라인에서 사용할 수 있도록 돕는 서비스이다.

> 온라인으로 공유할 수 있는 서비스

---

# 질문

1. 깃헙에서는 이미지까지만 업로드 할 수 있고, 영상이나 피피티는 안되는 건가요?

> 모든 파일을 관리할 수 있지만 변경사항 추적은 어려움. 또한 용량 제한이 있다(git-lfs라는 해결법이 있지만 제한적). 

2. 깃헙 초록색 네모칸을 채우는 방법은 커밋으로만 가능한가요? 자주 커밋을 할 필요가 혹시 없다면 다른 활동으로도 채울 수 있나 싶어서요!

> commit으로만 가능.

3. git 터미널에서 README.md 파일을 생성하고 푸시했는데 깃헙에서 자동으로 인식을 해주지 않았습니다. 내용이 없으면 자동인식을 해주지 않나요?

> 오타 혹은 내용이 없으면 인식하지 않는다.

---

# 실습 <자기소개서 프로젝트>

1. profile 폴더 생성

2. README.md
    - 간단한 자기소개 작성

3. 로컬저장소 설정(init)

4. commit

5. GitHub로 넘어와서 원격저장소 생성 => 이름은 username으로

6. 로컬에 원격저장소 설정

7. push

---

# .gitignore

- 공유를 하면 안되는 정보들이 저장되지 않도록 하는 방법.

- .gitignore 파일을 만든 후 이 안에 해당하는 파일명(또는 폴더, 확장자)를 작성하면 해당하는 파일들은 Git에 commit되지 않는다.

- 일반적으로 언어별 gitignore 파일을 만들어주는 사이트가 존재한다.
[gitignore 파일 생성 사이트](https://www.toptal.com/developers/gitignore/)

---

# push error

- GitHub(웹)상 History와 로컬의 History가 다를 경우 로컬에서 push할 때 에러가 발생한다.

- 해당 상황이 발생 시 History를 맞춰줘야 함. 이 때 사용하는 명령어가 `git pull`

---

# Branch

1. 독립적인 작업흐름을 만들고 정리

2. 여러 분기를 만들어서 여러 인원이 수정을 한 후 해당 내용을 main branch에 merge하는 방식으로 하나의 큰 프로젝트를 진행할 수 있다.

3. 주요 명령어

    - git branch 이름 : 해당 이름을 가진 branch 생성
    
    - git swich(checkout) 이름 : 해당 branch로 이동
    
    - git merge 이름 : main branch에서 실행하게 되면 main branch와 지정한 이름의 branch가 merge됨. 이 때 두 가지 방식이 있음(main과 해당 branch가 둘 다 수정 / 해당 branch만 수정)
    
    - git branch -d 이름 : 해당 branch 삭제

---

# Fork

- 다른 사람의 프로젝트를 가져오는 경우 `git clone 주소`로 받아올 수 있다. 하지만 이는 해당 프로젝트에 commit 권한이 없는 상태로 복사가 된다.

- 협업을 하는 경우에 사용할 수 있다.

- 해당 프로젝트를 fork한 후 내 GitHub에 추가된 Repository를 clone한 후 해당 폴더에서 작업을 한 후 commit, push한다.

- GitHub에서 **Pull requests**를 이용하여 해당 프로젝트에 merge 요청을 보낼 수 있다.

- merge의 경우 해당 프로젝트를 관리하고 있는 사람이 허가/취소 할 수 있다.

---

# Git에 대해 참고할만한 문서

[Git](https://git-scm.com/book/ko/v2)

[누구나 쉽게 이해할 수 있는 Git 입문](https://backlog.com/git-tutorial/kr/intro/intro1_1.html)