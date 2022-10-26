'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2022-10-26 12:21:28
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as data
import os
import pandas as pd

# Read the dataset
data_dir = 'https://raw.githubusercontent.com/bdfd/Project_01-Credit-Card-Approval-Prdiction/main/display%20demo/processed%20dataset.csv'

# Upload dataset into data frame
df = pd.read_csv(data_dir, encoding = 'utf-8')
# print(df.head(3))
df = data.encode(df)
X_train, X_test, y_train, y_test = data.split(df)
print(X_train.shape)
print(X_train)
data.model_evaluate(X_train, X_test, y_train, y_test)
