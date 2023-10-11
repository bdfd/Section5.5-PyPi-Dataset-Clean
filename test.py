'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-10-11 14:07:20
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
df = pd.read_csv("https://raw.githubusercontent.com/bdfd/Awesome_Dataset_Colltector/main/dataset%20collection/4.0%20github%20project/Section6.Project01/Car_Cleaned_Data.csv")

limit_number = 10
column = 'company'
list_1 = exe.filtered_value_list(df, column, limit_number)[0]
list_2 = exe.filtered_value_list(df, column, limit_number)[1]
print(list_1, list_2)
