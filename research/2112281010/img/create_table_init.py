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