'''
Date         : 2023-11-13 12:33:16
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-11-13 15:45:56
LastEditors  : BDFD
Description  : 
FilePath     : \unittest\graph\data_analysis_graph\discrete_feature_graph.py
Copyright (c) 2023 by BDFD, All Rights Reserved. 
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
numerical_features_list = exe.eda.numer

# visualise the numerical variables
print(df[numerical_features_list].head())
discrete_feature_list = exe.graph.discrete_feature_graph(
    df, target_feature, numerical_features_list)
