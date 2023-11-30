'''
Date         : 2023-11-30 10:17:35
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-11-30 12:41:37
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2023 by BDFD, All Rights Reserved. 
'''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import execdata as exe
# Sample DataFrame
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Yellow'],
    'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small'],
    'Shape': ['Square', 'Circle', 'Triangle', 'Circle', 'Square']
}
df = pd.DataFrame(data)

# Select columns to one-hot encode
categorical_features = ['Color', 'Size']

ohe = exe.eda.fit_one_hot_encode(df, categorical_features)

df_transformed = exe.eda.transform_one_hot_encode(
    df, categorical_features, ohe)
print("Transformed DataFrame:\n", df_transformed)

# Concatenate the encoded DataFrame with the original DataFrame, dropping the original categorical columns
result_df = exe.format.concat_transformed_df(
    df, df_transformed, categorical_features)

print("Original DataFrame:\n", df)
print("After Encoder Combined DataFrame:\n", result_df)
print("After Encoder Original DataFrame:\n", df)

# Inverse transform to revert back to the original form
df_reverted = exe.eda.inverse_one_hot_encode(
    df_transformed, categorical_features, ohe)
print("Reverted DataFrame:\n", df_reverted)
after_revert_result_df = exe.format.concat_inversed_df(
    result_df, df_reverted, df_transformed)
print("After Reverted Result DataFrame:\n", after_revert_result_df)
