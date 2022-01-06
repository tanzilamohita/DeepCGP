# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 1/6/2022
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

C7AIR_raw = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                       'C7AIR_PredictionAccuracy_Raw_BayesB.csv', low_memory=False)
C7AIR_comp1 = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_1_BayesB.csv', low_memory=False)
C7AIR_comp2 = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_2_BayesB.csv', low_memory=False)

C7AIR_raw_mean = C7AIR_raw.mean(axis=0)
C7AIR_comp1_mean = C7AIR_comp1.mean(axis=0)
C7AIR_comp2_mean = C7AIR_comp2.mean(axis=0)

# set height of bar
C7AIR_raw_list = list(C7AIR_raw_mean)
C7AIR_comp1_list = list(C7AIR_comp1_mean)
C7AIR_comp2_list = list(C7AIR_comp2_mean)

# for calculating Relative Accuracy
# set height of bar
# C7AIR_raw_list = list(C7AIR_raw_mean*100)
# C7AIR_comp1_list = list(C7AIR_comp1_mean*100)
# C7AIR_comp2_list = list(C7AIR_comp2_mean*100)

compression_level = ['0%', '57%', '94%']
accuracy_data = [C7AIR_raw_list, C7AIR_comp1_list, C7AIR_comp2_list]
accuracy_df = pd.DataFrame(np.column_stack(accuracy_data))
accuracy_df.columns = list(compression_level)
accuracy_df = accuracy_df.transpose()
accuracy_df.columns = ['Plant Height']
accuracy_df = np.array(accuracy_df)
# accuracy_df = accuracy_df/accuracy_df[0]*100 # for calculating Realtive Accuracy
accuracy_df = pd.DataFrame(accuracy_df, columns=['Plant Height'], index=compression_level)
print(accuracy_df)
# # accuracy_df.to_csv('../../../Data/C7AIR/C7AIR_Prediction_Accuracy/'
# #                         'C7AIR_PredictionAccuracy.csv')
#
accuracy_df.plot(kind="bar", figsize=(12, 8), rot=0)
# plt.title("Prediction Accuracy for C7AIR", fontsize=24)
plt.title("BayesB Prediction Accuracy for C7AIR", fontsize=24)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)
# plt.ylabel("Relative Accuracy (%)", fontsize=18)
plt.legend(loc='upper right', fontsize=12)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                       'C7AIR_BayesB_PredictionAccuracy.png')
# plt.savefig('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
#                        'C7AIR_PredictionRelativeAccuracy.png')
plt.show()

