import glob
import shutil as sh
import os
import random
import numpy as np

class file_processing():
    def __init__(self ,**kwargs):
        self.from_path = kwargs['from_path']
        self.to_path = kwargs['to_path']
    
    def find_file_path_lv1(self):
        ### 대상 파일 경로 확인 ###
        file_list = glob.glob(self.from_path + '/*' )
        print('File list lv1')
        for file in file_list:
            print(file)
        print('=' * 40)
    
    def find_file_path_lv2(self):
        ### 대상 파일 경로 확인 ###
        file_list = glob.glob(self.from_path + '/**/*' )
        print('File list lv2')
        for file in file_list:
            print(file)
        print('=' * 40)
    
    
    def find_file_path_lv3(self):
        file_list = glob.glob(self.from_path + '/**/**/*' )
        print('File list lv3: samples(', file_list[-1], ')')
        # for file in file_list:
        #     print(file)
        # print('=' * 40)
        self.file_list = np.array(file_list)
    
    def file_move(self, leng=0): ### 대상 파일 이동 ###
        idxs = np.arange(0, len(self.file_list)).tolist()
        picking = random.sample(idxs, leng)
        
        for file in self.file_list[picking]:
            file_name = file.split('/')[-1] # file_name = 파일명.type
            lv2_folder_name = file.split('/')[-2] # file_name = 파일명.type
            lv1_folder_name = file.split('/')[-3] # file_name = 파일명.type

            subfolder = '/'+self.to_path + '/' + lv1_folder_name +'/' + lv2_folder_name

            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            sh.move(file ,self.to_path + '/' + lv1_folder_name +'/' + lv2_folder_name+'/'+file_name)
            print(file_name, ' is move success.') 

    
    def file_copy(self, leng): ### 대상 파일 복사 ###
        idxs = np.arange(0, len(self.file_list)).tolist()
        picking = random.sample(idxs, leng)
        
        for file in self.file_list[picking]:
            file_name = file.split('/')[-1] # file_name = 파일명.type
            lv2_folder_name = file.split('/')[-2] # file_name = 파일명.type
            lv1_folder_name = file.split('/')[-3] # file_name = 파일명.type

            subfolder = '/'+self.to_path + '/' + lv1_folder_name +'/' + lv2_folder_name

            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            sh.copy(file ,self.to_path + '/' + lv1_folder_name +'/' + lv2_folder_name+'/'+file_name)
            print(file_name, ' is copy success.') 

from_path = '/home/joker1251/Desktop/owen/DataAnalysis_Science/DS_Master_21/Data/safety_class_dataset'
to_path = '/home/joker1251/Desktop/owen/DataAnalysis_Science/DS_Master_21/Data/safety_class_testset'

f = file_processing(from_path=from_path ,to_path = to_path)
f.find_file_path_lv3()
f.file_move(leng=2000)