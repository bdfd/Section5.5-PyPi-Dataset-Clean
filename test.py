'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-10-30 13:25:04
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as exe
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
import os

df = pd.read_csv(
    'https://raw.githubusercontent.com/bdfd/Section6.Project02-Telco_Customer_Churning_Prediction/main/1.0%20dataset/Telco-Customer-Churn.csv', encoding='utf-8')
# df2 = pd.read_csv(os.path.join(data_path,dataset_url_2), encoding = 'utf-8')

target_variable = 'Churn'
print(df[target_variable].value_counts())
print(' ')
majornity_target_value = 'No'
target_value_percentage = exe.majority_target_variable(
    df, target_variable, majornity_target_value)
print(target_value_percentage)
