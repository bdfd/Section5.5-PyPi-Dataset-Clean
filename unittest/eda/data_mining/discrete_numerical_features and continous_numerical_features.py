import pandas as pd
import numpy as np
import execdata as exe

df = pd.read_csv(
    'https://raw.githubusercontent.com/bdfd/Section6.Project04OP-Heart_Attack_Prediction/main/1.0%20dataset/Heart_Attack_Prediction.csv', encoding='utf-8')
new_column_names = {'output': 'Disease'}
df = df.rename(columns=new_column_names, level=0)
target_feature = 'Disease'
numerical_features_list = exe.eda.numerical_features_list(df)
# print(numerical_features_list)
discrete_numerical_features = exe.eda.discrete_numerical_feature_list(
    df, numerical_features_list)
continous_numerical_features = exe.eda.continuous_numerical_feature_list(
    df, numerical_features_list)
