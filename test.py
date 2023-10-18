'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-10-18 15:27:47
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as exe
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
import os

df = pd.read_csv(
    'https://raw.githubusercontent.com/bdfd/Section6.Project01-Car-Price-Predictor/Pickle-Demo/dataset/Car_Munged_Data.csv', encoding='utf-8')
# df2 = pd.read_csv(os.path.join(data_path,dataset_url_2), encoding = 'utf-8')
df = df.iloc[:, 1:]
print(df)

target_variable = 'Price'
X, y = exe.sep(df, target_variable)

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_transformation = make_column_transformer((OneHotEncoder(categories=ohe.categories_),
                                                ['name', 'company', 'fuel_type']),
                                                remainder='passthrough')

model = LinearRegression()
test_size, random_state = exe.sample_comparsion(X, y, column_transformation, model)
