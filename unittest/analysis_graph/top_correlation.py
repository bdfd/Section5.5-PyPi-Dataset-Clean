'''
Date         : 2022-10-26 11:24:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-11-02 16:40:25
LastEditors  : BDFD
Description  : 
FilePath     : \top_correlation.py
Copyright (c) 2022 by BDFD, All Rights Reserved. 
'''

import numpy as np
import pandas as pd
import execdata as exe

dataset_url = 'https://raw.githubusercontent.com/bdfd/Section6.Project02-Telco_Customer_Churning_Prediction/main/1.0%20dataset/Telco-Customer-Churn.csv'
df = pd.read_csv(dataset_url, encoding='utf-8')
target_feature = 'Churn'
cat_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
new_cat_features = exe.sort_categorical_feature(cat_features)
feature_le = exe.fit_label_encode(df, new_cat_features)
transformed_cat_df = exe.transform_label_encode(
    df, new_cat_features, feature_le)
top_correlation_list = exe.top_correlation(df, target_feature)
print(top_correlation_list)
