# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 3/12/2021
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hdra_raw = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                       'HDRA_PredictionAccuracyProcessed_Raw.csv', low_memory=False)
hdra_comp1 = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_1.csv', low_memory=False)
hdra_comp2 = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_2.csv', low_memory=False)
hdra_comp3 = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_3.csv', low_memory=False)

# hdra_raw_mean = hdra_raw.mean(axis=0)
# hdra_comp1_mean = hdra_comp1.mean(axis=0)
# hdra_comp2_mean = hdra_comp2.mean(axis=0)
# hdra_comp3_mean = hdra_comp3.mean(axis=0)
#
hdra_raw_mean = hdra_raw.mean(axis=0)[0:9]
hdra_comp1_mean = hdra_comp1.mean(axis=0)[0:9]
hdra_comp2_mean = hdra_comp2.mean(axis=0)[0:9]
hdra_comp3_mean = hdra_comp3.mean(axis=0)[0:9]

# hdra_raw_mean = hdra_raw.mean(axis=0)[9:18]
# hdra_comp1_mean = hdra_comp1.mean(axis=0)[9:18]
# hdra_comp2_mean = hdra_comp2.mean(axis=0)[9:18]
# hdra_comp3_mean = hdra_comp3.mean(axis=0)[9:18]

# # set width of bar
barWidth = 0.25
# fig = plt.subplots(figsize=(12, 8))

# set height of bar
hdra_raw_list = list(hdra_raw_mean*100)
hdra_comp1_list = list(hdra_comp1_mean*100)
hdra_comp2_list = list(hdra_comp2_mean*100)
hdra_comp3_list = list(hdra_comp3_mean*100)

# hdra_raw_list = [hdra_raw_mean[0]*100, hdra_comp1_mean[0]*100, hdra_comp2_mean[0]*100, hdra_comp3_mean[0]*100]
# hdra_comp1_list = [hdra_raw_mean[1]*100, hdra_comp1_mean[1]*100, hdra_comp2_mean[1]*100, hdra_comp3_mean[1]*100]
# hdra_comp2_list = [hdra_raw_mean[2]*100, hdra_comp1_mean[2]*100, hdra_comp2_mean[2]*100, hdra_comp3_mean[2]*100]
# hdra_comp3_list = [hdra_raw_mean[3]*100, hdra_comp1_mean[3]*100, hdra_comp2_mean[3]*100, hdra_comp3_mean[3]*100]

# print(list(hdra_raw)[1:5])
# print(hdra_raw_list)
# print(hdra_comp1_list)
# print(hdra_comp2_list)
# print(hdra_comp3_list)
# compression_level = ['Original', 'Compression_1', 'Compression_2', 'Compression_3']
compression_level = ['0%', '57%', '93%', '98%']
accuracy_data = [hdra_raw_list, hdra_comp1_list, hdra_comp2_list, hdra_comp3_list]
accuracy_df = pd.DataFrame(np.column_stack(accuracy_data))
# accuracy_df = accuracy_df
#accuracy_df.columns = list(hdra_raw)[1:10]
accuracy_df.columns = list(compression_level)
accuracy_df = accuracy_df.transpose()
accuracy_df.columns = list(hdra_raw)[1:10]
# accuracy_df.columns = list(hdra_raw)[10:19]
accuracy_df.columns = accuracy_df.columns.str.replace('.', ' ')
print(accuracy_df)
# accuracy_df.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                        'HDRA_PredictionAccuracy.csv')

accuracy_df.plot(kind="bar", figsize=(15, 10), rot=0, width=0.8)
plt.title("AutoRF Prediction Accuracy for HDRA (Traits 1-9)", fontsize=24)
# plt.title("Prediction Accuracy for HDRA (Traits 10-18)", fontsize=24)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.legend(loc='lower right', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('../HDRA_Data/HDRA_Prediction_Accuracy/'
                       'HDRA_PredictionAccuracy_1-9.png')
# plt.savefig('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                        'HDRA_PredictionAccuracy_10-18.png')
plt.show()

