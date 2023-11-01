'''
Date         : 2023-10-11 13:01:26
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-11-01 14:07:22
LastEditors  : BDFD
Description  : 
FilePath     : \test2.py
Copyright (c) 2023 by BDFD, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import execdata as exe
import os
# Alternative of Reading the dataset
# pwd = os.getcwd()
# data_dir = os.path.join(pwd, '50_Startups.csv')
# df = pd.read_csv(data_dir, encoding = 'utf-8')
# Read the dataset - switch to second link if first one not work, OP mean On Progressing

df = pd.read_csv(
    'https://raw.githubusercontent.com/bdfd/Section6.Project02-Telco_Customer_Churning_Prediction/main/1.0%20dataset/Telco-Customer-Churn.csv')
df.head()
# Total charges are in object dtype so convert into Numerical feature
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(
    df['tenure']*df['MonthlyCharges'])
df.drop(columns=['customerID'], inplace=True)
# print(df.head())
# print(df.isnull().sum())
cat_features = {feature for feature in df.columns if df[feature].dtypes == 'O'}
# print(len(cat_features))
print(df)
new_cat_features = exe.sort_categorical_feature(cat_features)
print(new_cat_features)
feature_le = exe.fit_label_encode(df, new_cat_features)
# print('0', cat_features[0])
# print('1', feature_le[0], 'with', feature_le[0].classes_)
# print('2', df['Churn'].unique())
new_df = exe.transform_label_encode(df, new_cat_features, feature_le)
# print(df)
print(new_df)
# print(df)
inverse_df = exe.inverse_label_encode(new_df, new_cat_features, feature_le)
print(inverse_df)
