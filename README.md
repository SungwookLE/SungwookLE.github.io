# SungwookLE 홈페이지

안녕하세요.

블로그는 [GitHub Pages](https://pages.github.com/)와 [Jekyll](https://jekyllrb.com/)을 사용하여 퍼블리싱하였으며, 마크다운 문법을 사용하여 글을 작성합니다.

----
## 📖 목차
- [환경구성](#️-환경구성)
  1. [프로젝트 clone](#프로젝트-clone)
  1. [Jekyll 설치](#jekyll-설치)
  1. [로컬에 블로그 실행](#로컬에-블로그-실행)

- [블로그 포스팅](#️-블로그-포스팅)
  1. [Typora 설치 (옵션)](#typora-설치-옵션)
  1. [글-작성](#글-작성)
  1. [commit과 push](#commit과-push)
  1. [Pull request 요청](#Pull-request-요청)
  
----
  
## ⚙️ 환경구성
### 프로젝트 clone
터미널에서 `git clone https://github.com/oy-alldev/oy-alldev.github.io.git` 명령어를 입력하여 프로젝트를 clone 받습니다.

만약 위 명령어를 입력했을 때 git을 찾을 수 없다는 오류가 발생한다면, git 공식 사이트([Windows](https://git-scm.com/download/win), [macOS](https://git-scm.com/download/mac))에서 git을 설치한 뒤 터미널을 재실행하여 재시도해봅니다.

### Jekyll 설치
터미널에서 `sudo gem install jekyll bundler` 명령어를 입력하여 `jekyll`과 `bundler`를 설치합니다.

만약 잘 되지 않는다면 공식 사이트([Windows](https://jekyllrb.com/docs/installation/windows/), [macOS](https://jekyllrb.com/docs/installation/macos/))에서 차근차근 따라하면 됩니다.


### 로컬에 블로그 실행
터미널에서 프로젝트 경로로 이동한 뒤 하기 명령어들을 실행하여 로컬에 블로그를 띄워줍니다.
```shell
# 패키지 설치
$ bundle install
# 블로그 서버 실행
$ jekyll serve 안되면 $bundle exec jekyll serve
```
위 과정에서 오류가 없었다면, 브라우저를 열어 [http://127.0.0.1:4000/](http://127.0.0.1:4000/)로 접속 시 로컬에서 블로그가 실행되는 것을 볼 수 있습니다.

#### 모든 설정이 완료되었습니다! 🎉
앞으로는 `jekyll serve` 명령어를 실행하는 것만으로도 손쉽게 블로그를 실행할 수 있습니다.

----

## ✍️ 블로그 포스팅

### Typora 설치 (옵션)
마크다운 제너레이터 없이도 마크다운을 잘 작성한다면 설치하지 않아도 무관하나, 우리는 도구를 사용하는 영장류이기 때문에 툴을 사용하면 좋습니다.
[Typora](https://typora.io/) 공식 사이트에 접속하여 프로그램을 설치합니다.

### 글 작성
1. 작성하고자하는 폴더로 이동합니다. (Research: `research` / Algorithm: `algorithm` / Day: `day`)
1. 현재시간 기준으로 `yyyymmddhhMM` 폴더를 생성합니다. (다른 폴더와 동명이지만 않으면 됩니다.)
1. 생성한 폴더 내에서 `Typora`나 메모장 등을 사용하여 마크다운 문서를 작성합니다. 포스트 내 이미지나 파일 첨부가 필요한 경우, 하위에 `img` 혹은 `file` 등의 폴더를 생성하여 넣어주는 편이 깔끔합니다.
1. 블로그 서버가 실행되어 있는 경우 브라우저에서 실시간으로 포스트 미리보기가 가능합니다.

### commit과 push
1. 터미널에서 프로젝트 경로로 이동합니다.
1. `git add .` 명령어를 통해 추가/수정 된 전체 파일을 changes list에 추가해줍니다. (일부 파일만 추가하고싶은 경우 `git add [파일명]` 명령어를 통해 일일이 추가할 수 있습니다.)
1. `git commit -m "[커밋 메시지]"` 명령어를 통해 커밋 메시지를 입력해줍니다. 커밋 메시지는 컨벤션에 맞게 작성합니다.
1. `git push -u origin master` 명령어를 통해 코드를 푸시합니다.

### Pull request 요청
1. github에 로그인하여 [repository](https://github.com/SungwookLE/SungwookLE.github.io)에 접속합니다.
1. 상단의 `Pull requests`를 눌러 좀 전에 push한 프로젝트를 `master`에 머지할 수 있도록 PR을 생성합니다.
1. 1명 이상의 팀원이 approve하여 merge하면, 몇 분 내로 [블로그](https://sungwookle.github.io/)에 글이 게재됩니다!

---- 
