'''
Date         : 2024-04-04 16:14:14
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2024-04-04 17:45:39
LastEditors  : <BDFD>
Description  : 
FilePath     : \unittest\eda\data_mining\feature_list.py
Copyright (c) 2024 by BDFD, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import execdata as exe
df = pd.read_csv(
    'https://raw.githubusercontent.com/bdfd/Section6.Project03-House_Price_Prediction/main/1.0%20dataset/train.csv', encoding='utf-8')
target_feature = 'SalePrice'
print(
    f'the dataset_1 size is {df.shape} and target feature is {target_feature}')
# list of numerical variables
numerical_features = exe.eda.numerical_features_list(df)
print(numerical_features)

categorical_features = exe.eda.categorical_features_list(df)
print(categorical_features)
