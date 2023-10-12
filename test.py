'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-10-12 15:48:14
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
df = pd.read_csv("https://raw.githubusercontent.com/bdfd/Awesome_Dataset_Colltector/main/dataset%20collection/4.0%20github%20project/Section6.Project01/SecondaryProcessed_Car_data.csv")

categorical_features = ['name', 'company', 'fuel_type']


features = exe.column_indentify(df, categorical_features)

print(features)
