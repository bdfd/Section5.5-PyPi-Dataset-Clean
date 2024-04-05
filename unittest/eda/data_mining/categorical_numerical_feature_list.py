'''
Date         : 2024-04-04 17:44:38
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2024-04-05 11:03:14
LastEditors  : <BDFD>
Description  : 
FilePath     : \unittest\eda\data_mining\categorical_numerical_feature_list.py
Copyright (c) 2024 by BDFD, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import execdata as exe

df = pd.read_csv(
    'https://raw.githubusercontent.com/bdfd/Section6.Project04OP-Heart_Attack_Prediction/main/1.0%20dataset/Heart_Attack_Prediction.csv', encoding='utf-8')
new_column_names = {'output': 'Disease'}
df = df.rename(columns=new_column_names, level=0)
target_feature = 'Disease'
numerical_features_list = exe.eda.numerical_features_list(df)
# print(numerical_features_list)
categorical_numerical_features = exe.eda.categorical_numerical_feature_list(
    df, numerical_features_list)
continous_numerical_features = exe.eda.continuous_numerical_feature_list(
    df, numerical_features_list)

exe.graph.categorical_numerical_feature_vs_target_graph(
    df, categorical_numerical_features, target_feature)
