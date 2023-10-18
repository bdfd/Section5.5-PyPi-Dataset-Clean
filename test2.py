'''
Date         : 2023-10-11 13:01:26
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-10-18 15:11:27
LastEditors  : BDFD
Description  : 
FilePath     : \test2.py
Copyright (c) 2023 by BDFD, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import os
# Alternative of Reading the dataset
# pwd = os.getcwd()
# data_dir = os.path.join(pwd, '50_Startups.csv')
# df = pd.read_csv(data_dir, encoding = 'utf-8')
# Read the dataset - switch to second link if first one not work, OP mean On Progressing

df = pd.read_csv(
    'https://raw.githubusercontent.com/bdfd/Section6.Project01-Car-Price-Predictor/Pickle-Demo/dataset/Car_Munged_Data.csv', encoding='utf-8')
# df2 = pd.read_csv(os.path.join(data_path,dataset_url_2), encoding = 'utf-8')
df = df.iloc[:, 1:]
print(df)
