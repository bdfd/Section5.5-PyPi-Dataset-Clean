'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2022-11-08 15:57:19
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
df = pd.read_csv("https://raw.githubusercontent.com/bdfd/Portfolio_Project_13OP-Cloth__Size_Prediction/main/dataset/cloth_size.csv")
print(df.head(3))
# drop Null small amount in the datasets
df = df.dropna(axis=0)
print(df.shape)
# we also drop the XXL record since we dont have a representative amount of this data
df = df[df['size'] != 'XXL']
print(df.shape)
# X = df.iloc[:,1:-1] # X value contains all the variables except labels -only if the prediction column is last one
# y = df.iloc[:,-1] # these are the labels
df_train, df_test = exe.split(df)
# rewrite the target variable
target_variable = 'size'
X_train, y_train, X_test, y_test = exe.sep(df_train, df_test, target_variable)
# we create the test train split first
dt_model = DecisionTreeClassifier(max_depth=8)
dt_model.fit(X_train, y_train)
dt_model.score(X_test, y_test)
y_predict = dt_model.predict(X_test)
exe.result_comparision(y_test, y_predict)