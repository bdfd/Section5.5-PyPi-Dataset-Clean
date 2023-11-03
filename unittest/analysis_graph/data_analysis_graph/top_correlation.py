'''
Date         : 2023-11-02 15:30:55
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-11-02 18:06:59
LastEditors  : BDFD
Description  : 
FilePath     : \top_correlation.py
Copyright (c) 2023 by BDFD, All Rights Reserved. 
'''

import numpy as np
import pandas as pd
import execdata as exe
# import execdata.analysis_graph

dataset_url = 'https://raw.githubusercontent.com/bdfd/Section6.Project02-Telco_Customer_Churning_Prediction/main/1.0%20dataset/Telco-Customer-Churn.csv'
df = pd.read_csv(dataset_url, encoding='utf-8')
target_feature = 'Churn'
cat_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
new_cat_features = exe.sort_categorical_feature(cat_features)
feature_le = exe.fit_label_encode(df, new_cat_features)
transformed_cat_df = exe.transform_label_encode(
    df, new_cat_features, feature_le)
top_correlation_list = exe.analysis_graph.top_correlation(
    df, target_feature)
print(top_correlation_list)
