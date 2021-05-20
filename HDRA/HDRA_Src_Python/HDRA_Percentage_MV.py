# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 13.10.2020
# =========================================


import numpy as np
import pandas as pd
import pathlib
from numpy import mean

# loading data
def HDRA_GenData():
    df = pd.read_csv('../HDRA_Data/HDRA_Raw/HDRA-G6-4-RDP1-RDP2-NIAS.AGCT.csv',
                     low_memory=False, na_values='N')
    df_transpose = df.transpose()
    df_range = df_transpose.iloc[1:, 0:]
    # gen_data = np.array(df_range)
    return df_range


print(HDRA_GenData().head())
total_mv = HDRA_GenData().isnull().sum().sum()
# Count total missing values in a DataFrame
print(" \nCount total missing values in a DataFrame: ", total_mv)
percent_missing = HDRA_GenData().isnull().sum() * 100 / len(HDRA_GenData())
print(percent_missing)
avg_mv_per = mean(percent_missing)
print(avg_mv_per)

pathlib.Path("../HDRA_Data/HDRA_MV_Percentage.txt")\
             .write_text("HDRA Total Missing Values: {} \n"
                         "HDRA Missing Value Percentage: {}"
             .format(total_mv, avg_mv_per))

# total = 0
# miss_perc = 0
# for i in range(HDRA_GenData().shape[1]):
#     # count number of rows with missing values
#     n_miss = HDRA_GenData()[[i]].isnull().sum()
#     perc = n_miss / HDRA_GenData().shape[0] * 100
#     print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
#     total += n_miss[i]
#     miss_perc += perc[i]
#
#
# print('Total missing values: ', total)
# print(miss_perc)
# avg_per = miss_perc/HDRA_GenData().shape[1]
# print('Total missing values percentage: ', avg_per)

# pathlib.Path("../HDRA_Data/HDRA_MV_Percentage.txt")\
#             .write_text("HDRA Total Missing Values: {} \n"
#                         "HDRA Missing Value Percentage: {}"
#             .format(total, avg_per))

