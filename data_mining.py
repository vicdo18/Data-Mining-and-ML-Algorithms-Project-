import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read multiple csv files
dir = ['S006.csv', 'S008.csv', 'S009.csv', 'S010.csv', 'S012.csv', 'S013.csv', 'S014.csv', 'S015.csv', 'S016.csv', 'S017.csv', 'S018.csv', 'S019.csv', 'S020.csv', 'S021.csv', 'S022.csv', 'S023.csv', 'S024.csv', 'S025.csv', 'S026.csv', 'S027.csv', 'S028.csv', 'S029.csv']

dfs = []
for file_name in dir:
    df = pd.read_csv(file_name)
    dfs.append(df)

#sub1 = pd.concat(dfs) 
# print(dfs[8].head())  # testing 

########################### STATS CALCULATION ################################

# for df in dfs:
#     # print(df.head())
#     # print(df.columns)
#     # print(df.describe())
#     # print(df.info())
#     # print(df.shape)
#     # print(df['value'].value_counts())
#     print("Mean of 'back_x' in", file_name, ":", df['back_x'].mean())
#     # print(df['value'].median())
#     # print(df['value'].std())
#     # print(df['value'].var())
#     # print(df['value'].mode())
#     # print(df['value'].min())
#     # print(df['value'].max())
#     # print(df['value'].corr())
#     # print(df['value'].cov())

for file_name, df in zip(dir, dfs):
    print("File:", file_name)
    print("Mean of 'back_x':", df['back_x'].mean())
    
    # You can add more statistics calculations here if needed
    print("Median of 'back_x':", df['back_x'].median())
    print("Standard deviation of 'back_x':", df['back_x'].std())
    print("Variance of 'back_x':", df['back_x'].var())
    print("Min of 'back_x':", df['back_x'].min())
    print("Max of 'back_x':", df['back_x'].max())
    #print("Correlation of 'back_x':", df['back_x'].corr())
    #print("Covariance of 'back_x':", df['back_x'].cov())

    print("-" * 50)

#save results to a csv file
for file_name, df in zip(dir, dfs):
    df_stats = pd.DataFrame()
    df_stats['Mean'] = df.mean()
    df_stats['Median'] = df.median()
    df_stats['Standard Deviation'] = df.std()
    df_stats['Variance'] = df.var()
    df_stats['Min'] = df.min()
    df_stats['Max'] = df.max()
    #df_stats['Correlation'] = df.corr()
    #df_stats['Covariance'] = df.cov()
    df_stats.to_csv(file_name + "_stats.csv") # save to a csv file

