'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2022-11-07 16:55:29
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as exe
import os
import numpy as np
import pandas as pd

# # Read the dataset
# data_dir = 'https://raw.githubusercontent.com/bdfd/Project_02-House_Model_Price_Prediction/main/display%20demo/processed%20dataset.csv'
# df = pd.read_csv(data_dir, encoding = 'utf-8')
# print(df.shape)

# a = 'income_category'
# # print(a.head(3))
# X, y = data.strat_split(df,a)
# print(X.shape)
# print(X.head(3))
data_dir = 'https://raw.githubusercontent.com/bdfd/Project_04OP-Wine_Category_Prediction/main/display%20demo/'
dataset_1 = 'test_x.csv'
dataset_2 = 'test_y.csv'
dataset_3 = 'train_x.csv'
dataset_4 = 'train_y.csv'
# Upload dataset into data frame
X_test = pd.read_csv(os.path.join(data_dir,dataset_1), encoding = 'utf-8')
y_test = pd.read_csv(os.path.join(data_dir,dataset_2), encoding = 'utf-8')
X_train = pd.read_csv(os.path.join(data_dir,dataset_3), encoding = 'utf-8')
y_train = pd.read_csv(os.path.join(data_dir,dataset_4), encoding = 'utf-8')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)
# print(scaled_X_test)
# print(scaled_X_train)
from sklearn.svm import SVC
svc_model = SVC()
y_train.drop(columns = y_train.columns[0], axis = 1, inplace= True)
y_test.drop(columns = y_test.columns[0], axis = 1, inplace= True)
y_train= np.array(y_train)
y_test= np.array(y_test)
# print(y_train)
svc_model.fit(scaled_X_train, y_train.ravel())
svc_model_predict = svc_model.predict(scaled_X_test)
y_test = y_test.ravel()
result = exe.algo_accuracy(y_test, svc_model_predict)
print(result)