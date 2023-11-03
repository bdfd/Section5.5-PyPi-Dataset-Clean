'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-11-03 14:02:08
LastEditors  : BDFD
Description  : 
FilePath     : \column_not_drop.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as exe
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


import execdata as exe

dataset_url = 'https://raw.githubusercontent.com/bdfd/Section6.Project02-Telco_Customer_Churning_Prediction/main/1.0%20dataset/Preprocessed_Data.csv'
df = pd.read_csv(dataset_url, encoding='utf-8')
print(df)
target_feature = 'Churn'
num_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
# print(num_features)
cat_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
# print(cat_features)
category_features = exe.data_preprocessing.column_identify(df,cat_features)
print(category_features)
new_cat_features = exe.data_preprocessing.sort_categorical_feature(
    cat_features)
feature_le = exe.data_preprocessing.fit_label_encode(df, new_cat_features)
transformed_cat_df = exe.data_preprocessing.transform_label_encode(
    df, new_cat_features, feature_le)
top10_cat_features = exe.analysis_graph.top_correlation(df, target_feature)
top10_cat_features = exe.data_preprocessing.sort_categorical_feature(
    top10_cat_features)
# print(top10_cat_features)
# print(len(top10_cat_features))
df = exe.data_preprocessing.column_not_drop(df, top10_cat_features)
print(df)
