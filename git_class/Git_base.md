# Git

## 1. Git이란?

- 분산 버전 관리 시스템으로 코드의 버전을 관리하는 도구이다.

- 컴퓨터 파일의 변경사항을 추적하고 여러 명의 사용자들 간에 파일들의 작업을 조율하는 역할을 한다.

- git 명령어

    - `git config --global init.defaultBranch main` : defaultBranch 명을 변경
    
    - `git config --global user.email "본인의 이메일"`: 본인의 이메일 정보를 Git이 인식할 수 있도록 코드 실행. GitHub에 가입한 정보로 하는 것이 연동이 편함
    
    - `git config --global user.name "본인의 이름"`: 위의 이메일과 마찬가지로 유저명을 인식할 수 있도록 하는 코드. GitHub에 가입한 정보로 하는 것이 좋음
    
    - `git init` : 현재 working directory를 git으로 관리하겠다는 명령어. 숨김파일로 .git이라는 폴더가 생긴다.
    
    - `git status` : 현재 working directory와 staging area의 상태를 보여준다.
    
    - `git add .(혹은 파일)` : 현재 working directory의 변경된 파일을 staging area로 저장하는 명령어.
    
    - `git commit -m '메시지'` : staging area에 모인 파일들의 버전을 남기는 명령어. 메시지의 경우 해당 commit에 대한 정보를 작성한다.
    
    - `git log` : commit한 기록을 보여준다. 다양한 옵션을 통해 보여주는 방식을 다르게 할 수 있다. (`git log -1` : 최근 1개 commit 목록을 보여줌, `git log -2 -oneline` : 최근 2개 commit 목록을 한 줄로 보여줌)
    
- Git은 데이터를 파일 시스템의 스냅샷으로 관리하고, 매우 크기가 작다. (파일이 달라지지 않으면 성능을 위해 파일을 새로 저장하지 않는다.)