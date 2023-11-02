'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-11-02 15:35:17
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''
import execdata as exe
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Example original data
# data = [['red', 'circle'],
#         ['blue', 'square'],
#         ['green', 'triangle']]

# # Example new data
# new_data = [['green', 'circle']]
# # Assuming data is a list of lists where each sub-list represents a row of the dataset

# label_encoder = exe.fit_label_encode(data)
# df_header = ['color', 'geometry']
# test_sample, reverse_sample = exe.transform_label_encode(
#     new_data, df_header, label_encoder)
# print(test_sample)
# print(type(test_sample))
# print(reverse_sample)
# print(type(reverse_sample))

import execdata as exe

dataset_url = 'https://raw.githubusercontent.com/bdfd/Section6.Project02-Telco_Customer_Churning_Prediction/main/1.0%20dataset/Telco-Customer-Churn.csv'
df = pd.read_csv(dataset_url, encoding='utf-8')
print(df)
