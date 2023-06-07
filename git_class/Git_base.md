# Git

## 1. Git이란?

- 분산 버전 관리 시스템으로 코드의 버전을 관리하는 도구이다.

- 컴퓨터 파일의 변경사항을 추적하고 여러 명의 사용자들 간에 파일들의 작업을 조율하는 역할을 한다.

## 2. Git의 특징

- Git은 데이터를 파일 시스템의 스냅샷으로 관리하고, 매우 크기가 작다. (파일이 달라지지 않으면 성능을 위해 파일을 새로 저장하지 않는다.)

## 3. git 명령어

    - `git config --global init.defaultBranch main` : defaultBranch 명을 변경
    
    - `git config --global user.email "본인의 이메일"`: 본인의 이메일 정보를 Git이 인식할 수 있도록 코드 실행. GitHub에 가입한 정보로 하는 것이 연동이 편함
    
    - `git config --global user.name "본인의 이름"`: 위의 이메일과 마찬가지로 유저명을 인식할 수 있도록 하는 코드. GitHub에 가입한 정보로 하는 것이 좋음
    
    - `git init` : 현재 working directory를 git으로 관리하겠다는 명령어. 숨김파일로 .git이라는 폴더가 생긴다.
    
    - `git status` : 현재 working directory와 staging area의 상태를 보여준다.
    
    - `git add .(혹은 파일)` : 현재 working directory의 변경된 파일을 staging area로 저장하는 명령어.
    
    - `git commit -m '메시지'` : staging area에 모인 파일들의 버전을 남기는 명령어. 메시지의 경우 해당 commit에 대한 정보를 작성한다.
    
    - `git log` : commit한 기록을 보여준다. 다양한 옵션을 통해 보여주는 방식을 다르게 할 수 있다. (`git log -1` : 최근 1개 commit 목록을 보여줌, `git log -2 -oneline` : 최근 2개 commit 목록을 한 줄로 보여줌)
    
    - `git remote add origin https://~~` : Git 원격저장소(remote) 추가(add). origin 이름으로 주소를 지정하겠다는 의미.
    
    - `git remote rm origin` : origin으로 지정된 원격저장소를 삭제하는 코드
    
    - `git remote -v` : 원격 저장소 목록을 조회하는 코드
    
    - `git push origin main` : Git 원격 저장소로 보내는(push) 코드. origin으로 지정된 저장소에 추가. main 자리의 경우 BranchName이다.
    
## 3. 질문 내용 중 생각해야할 것들

- commit은 내가 기록해야할 버전이라고 판단되면 하는 것이 좋음.(작업 중 저장의 개념이 아님)
ex) 데이터 전처리 - 분석 - 시각화 >> 해당 단계별로 commit

- commit 메시지: 기본적으로 내가 작업한 내용을 잘 담는 것이 중요.
보통 프로젝트땐 CRUD(Create/Read/Update/Delete) 정도로 정리가 된다. 
[참고링크](https://blog.ull.im/engineering/2019/03/10/logs-on-git.html)

- 여러 개를 묶어서 commit해야 하는 경우 `git add a.txt b.txt c.txt`처럼 commit을 할 수 있다. 띄어쓰기가 있는 경우 따옴표로 묶어서 사용