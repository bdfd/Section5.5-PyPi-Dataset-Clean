'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-10-16 15:27:04
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as exe
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Read the dataset

# Upload dataset into data frame
df = pd.read_csv(
    "https://raw.githubusercontent.com/bdfd/Section6.Project01-Car-Price-Predictor/Pickle-Demo/dataset/Car_Munged_Data.csv")


target_variable = "Price"
df = df.iloc[:, 1:]
X, y = exe.sep(df, target_variable)
# def sep(df, target_variable):
#     X = df.drop(target_variable, axis=1)
#     y = df[target_variable]
#     return X, y
print(X.head())
print(y.head())

X_train, X_test = exe.split(X)
print(X_train.shape)
print(X_test.shape)

y_train, y_test = exe.split(y)
print(y_train.shape)
print(y_test.shape)

X_train_2, y_train_2, X_test_2, y_test_2 = exe.sep_split(df, target_variable)
print(X_train_2.shape)
print(X_test_2.shape)
print(y_train_2.shape)
print(y_test_2.shape)