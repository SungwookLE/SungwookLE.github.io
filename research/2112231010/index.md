---
layout: post
type: research
date: 2021-12-23 10:10
category: XingAPI+MySQL
title: XingAPI + MySQL을 이용한 코스피, 코스닥 Stock DB 자동화 
subtitle: Stock DB 구축 자동화
writer: 100
post-header: true
header-img: img/trading.png
hash-tag: [mysql, mysql.connector, backend, database, db, XingAPI, Trade, Stock]
use_math: true
---

# XingAPI + MySQL을 이용한 코스피, 코스닥 Stock DB 자동화 
> Author: [SungwookLE](joker1251@naver.com)  
> Date  : '21.12/23
> Following Lecture: [Python을 이용한 주가 백테스팅 시스템 구축하기](https://www.inflearn.com/course/python-%EC%A3%BC%EA%B0%80-%EB%B0%B1%ED%85%8C%EC%8A%A4%ED%8C%85/dashboard)
> XingAPI 설명서: [COM개발가이드](./img/02.COM개발가이드.pdf), [COM객체Reference](./img/06.COM객체Reference.pdf)
>> 1. XingAPI 사용
>> 2. XingAPI+MySQL(1)
>> 3. XingAPI+MySQL(2)
>> 4. XingAPI+MySQL(3)

## 1. XingAPI 사용

- XingAPI를 사용하기 위해서 `COM` 객체를 활용하여 미리 정의된 객체를 불러오고 API에 맞게 호출해주어야 한다.
- `win32com.client`, `pythoncom`을 이용하여 com 객체를 받아올 수 있다.
- `XingAPI` 개발문서를 읽으면 사용법을 확인할 수 있는데, 아래와 같이 `EventHandler`를 활용해서 받아오게 된다.
   - `session = win32com.client.DispatchWithEvents("XA_Session.XASession", loginEventHandler) #내가 만든 Handler 클래스를 다중 상속시키는 것이다.` 
   - 개인적으로 다중상속을 해서 내가 지정한 `def OnLogin`함수에 들어오면 `loginEventHandler.is_login = True` 클래스 공유변수를 True로 SET 해주는 부분이 재밌었다.

- 이벤트를 호출해서 다시 받아와야 하는 것이기 때문에, 정상적으로 호출될 때 까지 기다려주어야 한다. 기다리는 것은 아래 구문으로 구현하였다.
    - `while loginEventHandler.is_login == False: pythoncom.PumpWaitingMessages()`
- `XingAPI`의 `tr`을 보고 객체 안에 들어있는 데이터를 맞게 가져와야 하는데, 이는 개발 문서를 잘 읽어보면서 진행해야 하므로, 외우지 말고 필요할 때 검색하자
    - `hname = t8430_session.GetFieldData("t8430OutBlock", "hname", index)`

```python
from pwd import xing_credentials
import win32com.client
import pythoncom
import time

class loginEventHandler:
    is_login = False #클래스 변수, 클래스 인스턴스 간 공유 변수
    # print(dir(loginEventHandler)) #클래스의 메쏘드와 공유 변수 출력할 수 있다.

    def OnLogin(self, code, msg):
        self.instance_var = 0 #인스턴스 변수, 인스턴스 개별적으로 사용하는 독립적인 변수
        print(code, msg)
        print("로그인 완료")
        loginEventHandler.is_login = True
    
    def OnDisconnect(self):
        pass

class t8430eventHandler:
    is_called = False
    def OnReceiveData(self, tr):
        print("불러오기 완료")
        print(tr)
        t8430eventHandler.is_called = True

session = win32com.client.DispatchWithEvents("XA_Session.XASession", loginEventHandler) #내가 만든 Handler 클래스를 다중 상속시키는 것이다.
session.ConnectServer("hts.ebestsec.co.kr", 20001)
print(session.IsConnected())

if session.IsConnected():
    session.Login(xing_credentials["ID"], xing_credentials["password"], xing_credentials["cert_password"], 0, 0)

while loginEventHandler.is_login == False:
    pythoncom.PumpWaitingMessages()

t8430_session = win32com.client.DispatchWithEvents("XA_DataSet.XAQuery", t8430eventHandler)
t8430_session.ResFileName = 'C:\\eBEST\\xingAPI\\Res\\t8430.res'
t8430_session.SetFieldData("t8430InBlock", "gubun", 0 , 0)
t8430_session.Request(0)

while t8430eventHandler.is_called == False:
    pythoncom.PumpWaitingMessages()

count=t8430_session.GetBlockCount("t8430OutBlock")
print(count)

for index in range(count):
    hname = t8430_session.GetFieldData("t8430OutBlock", "hname", index)
    shcode = t8430_session.GetFieldData("t8430OutBlock", "shcode", index)
    expcode = t8430_session.GetFieldData("t8430OutBlock", "expcode", index)
    etfgubun = t8430_session.GetFieldData("t8430OutBlock", "etfgubun", index)
    uplmtprice = t8430_session.GetFieldData("t8430OutBlock", "uplmtprice", index)
    dnlmtprice = t8430_session.GetFieldData("t8430OutBlock", "dnlmtprice", index)
    jnilclose = t8430_session.GetFieldData("t8430OutBlock", "jnilclose", index)
    memedan = t8430_session.GetFieldData("t8430OutBlock", "memedan", index)
    recprice = t8430_session.GetFieldData("t8430OutBlock", "recprice", index)
    gubun = t8430_session.GetFieldData("t8430OutBlock", "gubun", index)
    print("{0}: {1}".format(hname, gubun))
```

## 2. XingAPI+MySQL(1)
- Xing에서 `t8430` 블럭을 가져와서 company_list(코스닥+코스피) 를 읽어왔고, my_sql에 table을 생성하여 저장해주었다(`commit`)

```python
session = win32com.client.DispatchWithEvents("XA_Session.XASession", loginEventHandler) #내가 만든 Handler 클래스를 다중 상속시키는 것이다.
session.ConnectServer("hts.ebestsec.co.kr", 20001)
print(session.IsConnected())
if session.IsConnected():
    session.Login(xing_credentials["ID"], xing_credentials["password"], xing_credentials["cert_password"], 0, 0)

while loginEventHandler.is_login == False:
    pythoncom.PumpWaitingMessages()

t8430_session = win32com.client.DispatchWithEvents("XA_DataSet.XAQuery", t8430eventHandler)

t8430_session.ResFileName = 'C:\\eBEST\\xingAPI\\Res\\t8430.res'
t8430_session.SetFieldData("t8430InBlock", "gubun", 0 , 0)
t8430_session.Request(0)

while t8430eventHandler.is_called == False:
    pythoncom.PumpWaitingMessages()

count=t8430_session.GetBlockCount("t8430OutBlock")

print(count)

#mysql 에 접속하기
connection = mysql.connector.connect(user = mysql_credentials["user"] , password = mysql_credentials["password"], host = mysql_credentials["host"])
cursor_a = connection.cursor(buffered = True)
# db 생성
cursor_a.execute("create schema backtest")

# table 생성, table 생성에 이름은 `를 붙여주어야 한다
sql = "CREATE TABLE `backtest`.`total_company_list` (`hname` VARCHAR(45) NOT NULL, `shcode` VARCHAR(20) NULL, `expcode` VARCHAR(45) NULL, `etfgubun` VARCHAR(5) NULL, `uplmtprice` INT NULL, `dnlmtprice` INT NULL, `jinlclose` INT NULL, `memeda` VARCHAR(45) NULL, `recprice` INT NULL, `gubun` VARCHAR(5) NULL)"
cursor_a.execute(sql)
cursor_a.execute("use backtest")

for index in range(count):
    hname = t8430_session.GetFieldData("t8430OutBlock", "hname", index)
    shcode = t8430_session.GetFieldData("t8430OutBlock", "shcode", index)
    expcode = t8430_session.GetFieldData("t8430OutBlock", "expcode", index)
    etfgubun = t8430_session.GetFieldData("t8430OutBlock", "etfgubun", index)
    uplmtprice = t8430_session.GetFieldData("t8430OutBlock", "uplmtprice", index)
    dnlmtprice = t8430_session.GetFieldData("t8430OutBlock", "dnlmtprice", index)
    jnilclose = t8430_session.GetFieldData("t8430OutBlock", "jnilclose", index)
    memedan = t8430_session.GetFieldData("t8430OutBlock", "memedan", index)
    recprice = t8430_session.GetFieldData("t8430OutBlock", "recprice", index)
    gubun = t8430_session.GetFieldData("t8430OutBlock", "gubun", index)
    cursor_a.execute("insert into total_company_list values ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(hname, shcode, expcode, etfgubun, uplmtprice, dnlmtprice, jnilclose, memedan, recprice, gubun))
    #print("insert into total_company_list values ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(hname, shcode, expcode, etfgubun, uplmtprice, dnlmtprice, jnilclose, memedan, recprice, gubun))

# commit하여 최종 저장
connection.commit()
```

## 3. XingAPI+MySQL(2)
- `t8413` 블럭을 이용하여 Xing에서 데이터를 가지고 왔다.
- 삼성전자(shcode: `005930`)의 일봉을 쭉 불러와서 mysql에 테이블을 만들어 저장하였다.

```python
from pwd import xing_credentials, mysql_credentials
import win32com.client
import pythoncom
import time
import mysql.connector

class loginEventHandler:
    is_login = False #클래스 변수, 클래스 인스턴스 간 공유 변수
    # print(dir(loginEventHandler)) #클래스의 메쏘드와 공유 변수 출력할 수 있다.

    def OnLogin(self, code, msg):
        self.instance_var = 0 #인스턴스 변수, 인스턴스 개별적으로 사용하는 독립적인 변수

        print(code, msg)
        print("로그인 완료")
        loginEventHandler.is_login = True
    
    def OnDisconnect(self):
        pass

class t8413eventHandler:
    is_called = False

    def OnReceiveData(self, tr):
        print("불러오기 완료")
        print(tr)
        t8413eventHandler.is_called = True

session = win32com.client.DispatchWithEvents("XA_Session.XASession", loginEventHandler) #내가 만든 Handler 클래스를 다중 상속시키는 것이다.
session.ConnectServer("hts.ebestsec.co.kr", 20001)
print(session.IsConnected())
if session.IsConnected():
    session.Login(xing_credentials["ID"], xing_credentials["password"], xing_credentials["cert_password"], 0, 0)

while loginEventHandler.is_login == False:
    pythoncom.PumpWaitingMessages()

#mysql
connection = mysql.connector.connect(user = mysql_credentials["user"], password=mysql_credentials["password"], host =mysql_credentials["host"], database ="backtest" )
cursor_a = connection.cursor(buffered=True)

#create table for sh005930 , 삼성전자
sql="""CREATE TABLE `backtest`.`sh005930` (
  `date` DATE NOT NULL,
  PRIMARY KEY (`date`),
  `open` INT NULL,
  `high` INT NULL,
  `low` INT NULL,
  `close` INT NULL,
  `jdiff_vol` INT NULL,
  `value` INT NULL,
  `jongchk` INT NULL,
  `rate` DOUBLE NULL,
  `pricechk` INT NULL,
  `ratevalue` INT NULL,
  `sign` VARCHAR(5) NULL);"""
  
## mysql에서 인덱스, 저장부터 순서대로 저장하기 map/unordered_map
# 클러스터 인덱스 (primary index) -> 저장을 할 때 사전과 같이: 순서대로(알파벳, 오름차순..) ->table 속성에서 PK 지정해주면 클러스터 인덱스가 됨
#                                                                                      또는, 코드에서 PRIMARY KEY (`column` 이름) 을 써주면 됨
# 비클러스터 인덱스 (보조 인덱스, secondary index) -> 책 뒤에 잇는 색인처럼, 뒤죽박죽 unordered_map

cursor_a.execute(sql)

cts_date = "start"
edate = "20211213"
shcode = "005930" #삼성전자의 shcode임

while cts_date != "":
    print("-------------------------------------------------------------------------")
    t8413_session = win32com.client.DispatchWithEvents("XA_DataSet.XAQuery", t8413eventHandler)
    t8413_session.ResFileName = 'C:\\eBEST\\xingAPI\\Res\\t8413.res'
    t8413_session.SetFieldData("t8413InBlock", "shcode", 0 , shcode) 
    t8413_session.SetFieldData("t8413InBlock", "gubun", 0 , 2) #2: 일봉 , #3: 주봉, #4: 월봉
    t8413_session.SetFieldData("t8413InBlock", "sdate", 0 , "20130101")
    t8413_session.SetFieldData("t8413InBlock", "edate", 0 , edate)

    """
    t8413_session.SetFieldData("t8413InBlock", "qrycnt", 0 , 500) #비압축 (비압축은 500개가 최대)
    t8413_session.SetFieldData("t8413InBlock", "comp_yn", 0 , "N") #비압축
    """
    t8413_session.SetFieldData("t8413InBlock", "qrycnt", 0 , 2000) #압축
    t8413_session.SetFieldData("t8413InBlock", "comp_yn", 0 , "Y") #압축

    t8413_session.Request(0)

    while t8413eventHandler.is_called == False:
        pythoncom.PumpWaitingMessages()

    t8413_session.Decompress("t8413OutBlock1")
    count = t8413_session.GetBlockCount("t8413OutBlock1")

    cts_date = t8413_session.GetFieldData("t8413OutBlock", "cts_date", 0)
    for index in range(count):
        date = t8413_session.GetFieldData("t8413OutBlock1", "date", index)
        open = t8413_session.GetFieldData("t8413OutBlock1", "open", index)
        high = t8413_session.GetFieldData("t8413OutBlock1", "high", index)
        low = t8413_session.GetFieldData("t8413OutBlock1", "low", index)
        close = t8413_session.GetFieldData("t8413OutBlock1", "close", index)
        jdiff_vol = t8413_session.GetFieldData("t8413OutBlock1", "jdiff_vol", index)
        value = t8413_session.GetFieldData("t8413OutBlock1", "value", index)
        jongchk = t8413_session.GetFieldData("t8413OutBlock1", "jongchk", index)
        rate = t8413_session.GetFieldData("t8413OutBlock1", "rate", index)
        pricechk = t8413_session.GetFieldData("t8413OutBlock1", "pricechk", index)
        ratevalue = t8413_session.GetFieldData("t8413OutBlock1", "ratevalue", index)
        sign = t8413_session.GetFieldData("t8413OutBlock1", "sign", index)

        # insert table data
        #print(date, open, high, low, close, jdiff_vol, value, jongchk, rate, pricechk, ratevalue, sign)
        sql = "insert into sh{} (date, open, high, low, close, jdiff_vol, value, jongchk, rate, pricechk, ratevalue, sign) values ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(shcode, date, open, high, low, close, jdiff_vol, value, jongchk, rate, pricechk, ratevalue, sign)
        cursor_a.execute(sql)

    print("Read:", count , "ea")
    if cts_date != "":
        print("연속데이터 있습니다: ", cts_date)
        edate = cts_date
    else:
        print("전부 조회 완료")

    # commit하여 저장
    connection.commit()

    time.sleep(1) # xingAPI 건당 제한 속도 1초에 1건
    t8413eventHandler.is_called = False #리퀘스트를 제대로 기다려주게 하기 위해 `while t8413eventHandler.is_called == False:` 루프마다 초기화
    print("-------------------------------------------------------------------------")
```

## 4. XingAPI+MySQL(3)
- 최종적으로 아래 코드를 이용하여 StockD DB를 생성하였다. (시작: 20110101 ~ 끝: 20211223)
- 저장한 테이블(company_list)를 이용하여, 모든 company의 shcode를 가져오고, shcode에 대응하는 일봉 데이터를 `t8413`에서 불러와 mysql에 저장하였다.
- mysql에서 인덱스, 저장부터 순서대로 저장하기 c++의 map/unordered_map와 비슷
    1. 클러스터 인덱스 (primary index) -> 저장을 할 때 사전과 같이: 순서대로(알파벳, 오름차순..) ->table 속성에서 PK 지정해주면 클러스터 인덱스가 됨. 또는, 코드에서 PRIMARY KEY (`column` 이름) 을 써주면 됨
    2. 비클러스터 인덱스 (보조 인덱스, secondary index) -> 책 뒤에 잇는 색인처럼, 뒤죽박죽 unordered_map

```python
from pwd import xing_credentials, mysql_credentials
import win32com.client
import pythoncom
import time
import mysql.connector

# total_company_list를 이용하여 개별종목 table 생성 자동화하기

class loginEventHandler:
    is_login = False #클래스 변수, 클래스 인스턴스 간 공유 변수
    # print(dir(loginEventHandler)) #클래스의 메쏘드와 공유 변수 출력할 수 있다.

    def OnLogin(self, code, msg):
        self.instance_var = 0 #인스턴스 변수, 인스턴스 개별적으로 사용하는 독립적인 변수

        print(code, msg)
        print("로그인 완료")
        loginEventHandler.is_login = True
    
    def OnDisconnect(self):
        pass

class t8413eventHandler:
    is_called = False

    def OnReceiveData(self, tr):
        print("불러오기 완료")
        print(tr)
        t8413eventHandler.is_called = True

session = win32com.client.DispatchWithEvents("XA_Session.XASession", loginEventHandler) #내가 만든 Handler 클래스를 다중 상속시키는 것이다.
session.ConnectServer("hts.ebestsec.co.kr", 20001)
print(session.IsConnected())
if session.IsConnected():
    session.Login(xing_credentials["ID"], xing_credentials["password"], xing_credentials["cert_password"], 0, 0)

while loginEventHandler.is_login == False:
    pythoncom.PumpWaitingMessages()

#mysql
connection = mysql.connector.connect(user = mysql_credentials["user"], password=mysql_credentials["password"], host =mysql_credentials["host"], database ="backtest" )
cursor_a = connection.cursor(buffered=True)

def create_table(shcode_input):
    cursor_a.execute("USE backtest")
    # create table for sh005930 , 삼성전자
    sql="""CREATE TABLE `backtest`.sh{} (
    `date` DATE NOT NULL,
    PRIMARY KEY (`date`),
    `open` INT NULL,
    `high` INT NULL,
    `low` INT NULL,
    `close` INT NULL,
    `jdiff_vol` INT NULL,
    `value` INT NULL,
    `jongchk` INT NULL,
    `rate` DOUBLE NULL,
    `pricechk` INT NULL,
    `ratevalue` INT NULL,
    `sign` VARCHAR(5) NULL);""".format(shcode_input)
    
    ## mysql에서 인덱스, 저장부터 순서대로 저장하기 map/unordered_map
    # 클러스터 인덱스 (primary index) -> 저장을 할 때 사전과 같이: 순서대로(알파벳, 오름차순..) ->table 속성에서 PK 지정해주면 클러스터 인덱스가 됨
    #                                                                                      또는, 코드에서 PRIMARY KEY (`column` 이름) 을 써주면 됨
    # 비클러스터 인덱스 (보조 인덱스, secondary index) -> 책 뒤에 잇는 색인처럼, 뒤죽박죽 unordered_map
    try:
        cursor_a.execute(sql)
    except:
        print("(Skip)Existed DB: ", shcode_input)
        return

    print("(New)Generate DB: ", shcode_input)

    cts_date = "start"
    edate = "20211223"
    shcode = shcode_input 

    while cts_date != "":
        print("-------------------------------------------------------------------------")
        t8413_session = win32com.client.DispatchWithEvents("XA_DataSet.XAQuery", t8413eventHandler)
        t8413_session.ResFileName = 'C:\\eBEST\\xingAPI\\Res\\t8413.res'
        t8413_session.SetFieldData("t8413InBlock", "shcode", 0 , shcode) 
        t8413_session.SetFieldData("t8413InBlock", "gubun", 0 , 2) #2: 일봉 , #3: 주봉, #4: 월봉
        t8413_session.SetFieldData("t8413InBlock", "sdate", 0 , "20110101")
        t8413_session.SetFieldData("t8413InBlock", "edate", 0 , edate)

        """
        t8413_session.SetFieldData("t8413InBlock", "qrycnt", 0 , 500) #비압축 (비압축은 500개가 최대)
        t8413_session.SetFieldData("t8413InBlock", "comp_yn", 0 , "N") #비압축
        """
        t8413_session.SetFieldData("t8413InBlock", "qrycnt", 0 , 2000) #압축
        t8413_session.SetFieldData("t8413InBlock", "comp_yn", 0 , "Y") #압축

        t8413_session.Request(0)

        while t8413eventHandler.is_called == False:
            pythoncom.PumpWaitingMessages()

        t8413_session.Decompress("t8413OutBlock1")
        count = t8413_session.GetBlockCount("t8413OutBlock1")

        cts_date = t8413_session.GetFieldData("t8413OutBlock", "cts_date", 0)
        for index in range(count):
            date = t8413_session.GetFieldData("t8413OutBlock1", "date", index)
            open = t8413_session.GetFieldData("t8413OutBlock1", "open", index)
            high = t8413_session.GetFieldData("t8413OutBlock1", "high", index)
            low = t8413_session.GetFieldData("t8413OutBlock1", "low", index)
            close = t8413_session.GetFieldData("t8413OutBlock1", "close", index)
            jdiff_vol = t8413_session.GetFieldData("t8413OutBlock1", "jdiff_vol", index)
            value = t8413_session.GetFieldData("t8413OutBlock1", "value", index)
            jongchk = t8413_session.GetFieldData("t8413OutBlock1", "jongchk", index)
            rate = t8413_session.GetFieldData("t8413OutBlock1", "rate", index)
            pricechk = t8413_session.GetFieldData("t8413OutBlock1", "pricechk", index)
            ratevalue = t8413_session.GetFieldData("t8413OutBlock1", "ratevalue", index)
            sign = t8413_session.GetFieldData("t8413OutBlock1", "sign", index)

            # insert table data
            #print(date, open, high, low, close, jdiff_vol, value, jongchk, rate, pricechk, ratevalue, sign)
            sql = "insert into sh{} (date, open, high, low, close, jdiff_vol, value, jongchk, rate, pricechk, ratevalue, sign) values ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(shcode, date, open, high, low, close, jdiff_vol, value, jongchk, rate, pricechk, ratevalue, sign)
            cursor_a.execute(sql)


        print("Read:", count , "ea")
        if cts_date != "":
            print("연속데이터 있습니다: ", cts_date)
            edate = cts_date
        else:
            print("전부 조회 완료")

        # commit하여 저장
        connection.commit()

        time.sleep(3.5) # xingAPI 건당 제한 속도 1초에 1건
        t8413eventHandler.is_called = False #리퀘스트를 제대로 기다려주게 하기 위해 `while t8413eventHandler.is_called == False:` 루프마다 초기화
        print("-------------------------------------------------------------------------")


cursor_list = connection.cursor(buffered=True)
cursor_list.execute("select shcode from total_company_list")

for shcode in cursor_list:
    create_table(shcode[0])
```

## 끝