'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2022-10-28 14:23:03
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as data
import os
import pandas as pd

# Read the dataset
data_dir = 'https://raw.githubusercontent.com/bdfd/Project_02-House_Model_Price_Prediction/main/display%20demo/processed%20dataset.csv'
df = pd.read_csv(data_dir, encoding = 'utf-8')
print(df.shape)

a = 'income_category'
# print(a.head(3))
X, y = data.strat_split(df,a)
print(X.shape)
print(X.head(3))

