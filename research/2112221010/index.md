---
layout: post
type: research
date: 2021-12-22 10:10
category: MySQL
title: MySQL 
subtitle: db 구축을 위한 mysql 스터디
writer: 100
post-header: true
header-img: img/
hash-tag: mysql, mysql.connector, backend, database, db]
use_math: true
---

- toc
{:toc}

# MySQL
> Author: [SungwookLE](joker1251@naver.com)  
> Date  : '21.12/22
> Following Lecture: [Python을 이용한 주가 백테스팅 시스템 구축하기](https://www.inflearn.com/course/python-%EC%A3%BC%EA%B0%80-%EB%B0%B1%ED%85%8C%EC%8A%A4%ED%8C%85/dashboard)
>> 1. 환경설정
>> 2. mysql_connector
>> 3. mysql_transaction

## 1. 환경설정
1. anaconda를 이용하여 32bit python3.6 가상환경 생성
2. xingApi 설치 (이베스트 증권, 32bit virenv 필요 이유), 공인인증서 필요
3. mysql server, workbench **2개** 설치
4. pip install mysql-connector-python: python API 활용
- tip: mysql을 사용하지 않을 때는 서비스를 중지시켜 불필요한 PC 리소스 소모를 방지시키자
    1. window PC
    - 실행->서비스->mysql->중지
    - 재사용할 경우에는 실행->서비스->mysql->시작/재시작

    2. ubuntu PC (`service` 명령어가 안될 경우 `systemctl` 사용)
    - 상태확인: `service mysql status`
    - 시작: `service mysql start`
    - 정지: `service mysql stop`
    - 재시작: `service mysql restart`

    - `ps -ef | grep mysql`
    - `ps -A | grep mysql`: 프로세스 중 mysql 정보 출력
    - `sudo pkill mysql`: 프로세스 종료(킬)

## 2. mysql_connector
- `mysql.connector`라는 객체를 생성한다. 이 때 `user, password, host` 정보를 입력
- `cursor_a`는 하나의 실행체(말 그대로 마우스 커서)라고 보면 된다.
- `connection.cursor.execute`는 sql문을 실행시키는 함수이다. 따라서, `execute("SQL문")`의 형태로 써주어야 한다.
- sql문이 실행된 결과는 `cursor`에 담긴다.

- 아래의 코드는 mysql API를 이용하여 mysql db에 접근하고 테이블 데이터를 가져오는 예시이다.

```python
import mysql.connector
from pwd import credentials #password파일은 비공개로 하기위해 해당 라인처럼 파일 관리

'''
방법1) database.table를 query문에 직접 입력
'''
# Connect with the MySQL Server
connection = mysql.connector.connect(user=credentials["user"], password = credentials["password"], host=credentials["host"])
# Get buffered cursors
cursor_a = connection.cursor(buffered = True)
# Query to get the table data
sql = "select * from backtest_db.table_test"
cursor_a.execute(sql)
print("방법1) database.table를 query문에 직접 입력")
for item in cursor_a:
    print(item)

'''
방법2) sql문을 이용하여 `use database`를 입력
'''
# Connect with the MySQL Server
connection = mysql.connector.connect(user=credentials["user"], password = credentials["password"], host=credentials["host"])

# Get buffered cursors
cursor_a = connection.cursor(buffered = True)
cursor_a.execute("use backtest_db")
# Query to get the table data
sql = "select * from table_test"
cursor_a.execute(sql)
print("방법2) sql문을 이용하여 `use database`를 입력")
for item in cursor_a:
    print(item)

'''
방법3) connect에서 database를 명기
'''
# Connect with the MySQL Server
connection = mysql.connector.connect(user=credentials["user"], password = credentials["password"], host=credentials["host"], database="backtest_db")
# Get buffered cursors
cursor_a = connection.cursor(buffered = True)
# Query to get the table data
sql = "select * from table_test"
cursor_a.execute(sql)
print("방법3) connect에서 database를 명기")
for item in cursor_a:
    print(item)
```

