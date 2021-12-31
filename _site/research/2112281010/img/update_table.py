import win32com.client
import pythoncom
import time
import mysql.connector

# total_company_list를 이용하여 개별종목 table 생성 자동화하기

class t8413eventHandler:
    is_called = False
    def OnReceiveData(self, tr):
        print("불러오기 완료")
        print(tr)
        t8413eventHandler.is_called = True

def update_table(shcode_input, connection, s_date, e_date):
    cursor_a = connection.cursor(buffered=True)
    cursor_a.execute("USE backtest")

    cts_date = "start"
    shcode = shcode_input 

    while cts_date != "":
        print("-------------------------------------------------------------------------")
        t8413_session = win32com.client.DispatchWithEvents("XA_DataSet.XAQuery", t8413eventHandler)
        t8413_session.ResFileName = 'C:\\eBEST\\xingAPI\\Res\\t8413.res'
        t8413_session.SetFieldData("t8413InBlock", "shcode", 0 , shcode) 
        t8413_session.SetFieldData("t8413InBlock", "gubun", 0 , 2) #2: 일봉 , #3: 주봉, #4: 월봉
        t8413_session.SetFieldData("t8413InBlock", "sdate", 0 , s_date)
        t8413_session.SetFieldData("t8413InBlock", "edate", 0 , e_date)

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
            try:
                cursor_a.execute(sql)
            except mysql.connector.errors.IntegrityError as err:
                print("중복 존재")

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