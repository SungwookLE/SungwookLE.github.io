from logging import log
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from main_ui import Ui_MainWindow
import datetime

import os
#상위 경로 폴더를 가져오기 위해 선언
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from lecture_practice.pwd import mysql_credentials, xing_credentials
from lecture_practice.update_table import update_table

#mysql
import mysql.connector

#XingAPI
import win32com.client
import pythoncom

class mysql_status:
    def check_mysql_status():
        msg = os.popen('sc query MySQL80').read()
        for tok in msg.split(' '):
            if tok == "RUNNING":
                is_on = True
                return is_on
        else:
            is_on = False
            return is_on
   
class loginEventHandler:
    is_login= False
    is_error= False

    is_pwd_wrong = False
    is_id_wrong =False
    is_cert_pwd_wrong = False

    def OnLogin(self, code, msg):
        print(code, msg)
        loginEventHandler.is_login = True
        if code != "0000":
           loginEventHandler.is_error=True
           if code == "5101":
           #     #ID 확인
               loginEventHandler.is_id_wrong = True
           elif code == "8004":
           #      #PWD 틀림
               loginEventHandler.is_pwd_wrong = True
           elif code == "2005":
           # 공동인증서 비번 틀림
               loginEventHandler.is_cert_pwd_wrong = True
        
    def reset_flag():
        loginEventHandler.is_login= False
        loginEventHandler.is_error= False
        loginEventHandler.is_pwd_wrong = False
        loginEventHandler.is_id_wrong =False
        loginEventHandler.is_cert_pwd_wrong = False

class mainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, debug_mode=False):
        super().__init__()
        self.setupUi(self)
        self.connection = False
        self.session = False
        self.db_login_button.clicked.connect(self.db_login)
        self.xing_login_button.clicked.connect(self.api_login)
        self.db_update_button.setEnabled(False)
        self.db_update_button.clicked.connect(self.update_database)

        if (mysql_status.check_mysql_status()):
            self.db_status_text.setText('ON')
        else:
            self.db_status_text.setText('OFF')

        if debug_mode ==True:
            self.db_id_edit.setText(mysql_credentials["user"])
            self.db_pwd_edit.setText(mysql_credentials["password"])
            self.db_ip_edit.setText(mysql_credentials["host"])
            self.db_schema_edit.setText("backtest")

            self.xing_id_edit.setText(xing_credentials["ID"])
            self.xing_pwd_edit.setText(xing_credentials["password"])
            self.cert_pwd_edit.setText(xing_credentials["cert_password"])

    def db_login(self):
        user = self.db_id_edit.text()
        password = self.db_pwd_edit.text()
        host = self.db_ip_edit.text()
        database = self.db_schema_edit.text()
        try:
            self.connection = mysql.connector.connect(user=user , password=password, host=host, database=database)
        except:
            # 정말 절대로 이런식으로 예외처리하면 안좋지만 시간관계상 이렇게 진행
            self.log_text("db 로그인 실패...")
        else:
            self.log_text("db 로그인 성공!!!")
            self.db_login_button.setEnabled(False)
        
        self.check_login_status()

    def api_login(self):
        self.session = win32com.client.DispatchWithEvents("XA_Session.XASession", loginEventHandler)
        self.session.ConnectServer("hts.ebestsec.co.kr", 20001)

        xing_id = self.xing_id_edit.text()
        xing_pwd = self.xing_pwd_edit.text()
        cert_pwd = self.cert_pwd_edit.text()

        if self.session.IsConnected():
            self.log_text("증권사 서버 연결완료, 로그인 진행 중...")
            self.session.Login(xing_id, xing_pwd, cert_pwd, 0, 0)

        while loginEventHandler.is_login == False:
            pythoncom.PumpWaitingMessages()
        
        if loginEventHandler.is_error == True:
            if loginEventHandler.is_id_wrong == True:
                self.log_text("Xing API: ID 틀림...")
            elif loginEventHandler.is_pwd_wrong == True:
                self.log_text("Xing API: PWD 틀림...")
            elif loginEventHandler.is_cert_pwd_wrong == True:
                self.log_text("Xing API: 공동인증서 PWD 틀림...")

        else:
            self.log_text("Xing API 로그인 성공!!!")
            self.xing_login_button.setEnabled(False)

        self.check_login_status()
        loginEventHandler.reset_flag()

    def log_text(self, msg):
        self.textEdit.append("[+]{} {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))

    def check_login_status(self):
        if (self.db_login_button.isEnabled() == False and self.xing_login_button.isEnabled() == False):
            self.log_text("DB, Xing API 로그인 완료되어 업데이트가 가능합니다.")
            self.db_update_button.setEnabled(True)

    def update_database(self):
        edate = datetime.date.today() - datetime.timedelta(1)  #하루 전날까지의 데이터만 업데이트      
        cursor_a = self.connection.cursor(buffered=True)
        cursor_a.execute("select date from sh005930 order by date desc limit 1")
        database_latest_date = cursor_a.fetchone()[0]
        if edate - database_latest_date >= datetime.timedelta(1):
            self.log_text("데이터베이스 업데이트를 시작합니다.")
            cursor_a.execute("select shcode from total_company_list")
            for shcode in cursor_a:
                update_table(shcode[0], self.connection, database_latest_date.strftime('%Y%m%d'), edate.strftime('%Y%m%d'))
                self.log_text("{} 종목 업데이트 완료".format(shcode[0]))
        else:
            self.log_text("이미 데이터베이스가 최신이거나 업데이트가 불가능합니다.")

app = QApplication(sys.argv)
test_window = mainWindow(debug_mode=True)
test_window.show()

app.exec_()

