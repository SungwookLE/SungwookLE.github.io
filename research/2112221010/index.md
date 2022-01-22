---
layout: post
type: research
date: 2021-12-22 10:10
category: MySQL
title: MySQL 
subtitle: db 구축을 위한 mysql 스터디
writer: 100
post-header: true
header-img: img/mysql.jpg
hash-tag: [mysql, mysql.connector, backend, database, db]
use_math: true
---

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

### 1-1. Tip: mysql을 사용하지 않을 때는 서비스를 중지시켜 불필요한 PC 리소스 소모를 방지시키자
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

- 또한, mysql에 접속 실패하거나 하였을 때 에러를 핸들링하는 방법도 있다.
- 해당 [docu](https://dev.mysql.com/doc/connector-python/en/connector-python-example-connecting.html)를 참고해보자 

```python
import mysql.connector
from mysql.connector import errorcode

try:
  cnx = mysql.connector.connect(user='scott',
                                database='employ')
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cnx.close()
```

## 3. mysql_transaction
- db를 `commit()`하거나, `rollback()`하는 것을 말함
- `commit()`은 내가 현재 `cursor`에서 작업한 것을 db에 최종 저장시키는 것을 말함
- `rollback()`은 `cursor`에서 작업한 내용을 되돌리는 것을 말한다.

```python
import mysql.connector
from pwd import credentials #password파일은 비공개로 하기위해 해당 라인처럼 파일 관리

# Connect with the MySQL Server
connection = mysql.connector.connect(user=credentials["user"], password = credentials["password"], host=credentials["host"])
# Get buffered cursors
cursor_a = connection.cursor(buffered = True)
# Query to get the table data
print("-------init-----------")
sql = "select * from backtest_db.table_test"
cursor_a.execute(sql)
for item in cursor_a:
    print(item)


#commit()은 저장하기이고, rollback()은 되돌리기임
print("\n-------delete-----------")
cursor_a.execute("delete from backtest_db.table_test")
 
connection.rollback() #cursor 작업 내용을 되돌리기
sql = "select * from backtest_db.table_test"
cursor_a.execute(sql)

for item in cursor_a:
    print(item)
    
connection.commit() #cursor 작업 내용을 db에 최종 저장하기
```

- transaction이 필요한 이유는, 은행 입출금을 할 때, 송금자와 수금자가 서로 돈 거래가 잘 db에 반영되었을 때 최종적으로 commit을 하는 것이 안정적이기 때문이다.

```python
try:
    cur.execute("update toss_table set balance ={} where user = {}".format(from_user_balance-mount, from_user))
    cur.execute("update toss_table set balance ={} where user = {}".format(to_user_balance+mount, to_user))
except:
    connection.rollback()
else:
    connection.commit()
```