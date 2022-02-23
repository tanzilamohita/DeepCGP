# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 7/9/2021
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

# processing Ranger Prediction Accuracy
C7AIR_raw_ranger = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                       'C7AIR_PredictionAccuracy_Raw.csv', low_memory=False)
C7AIR_comp1_ranger = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_1.csv', low_memory=False)
C7AIR_comp2_ranger = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_2.csv', low_memory=False)

# set height of bar
C7AIR_raw_mean_ranger = statistics.mean(list(C7AIR_raw_ranger.mean(axis=0)))
C7AIR_comp1_mean_ranger = statistics.mean(list(C7AIR_comp1_ranger.mean(axis=0)))
C7AIR_comp2_mean_ranger = statistics.mean(list(C7AIR_comp2_ranger.mean(axis=0)))

## Calculate Standard Deviation
C7AIR_raw_SD_RF = np.std(list(C7AIR_raw_ranger.mean(axis=0)*100), dtype=np.float64)
C7AIR_comp1_SD_RF = np.std(list(C7AIR_comp1_ranger.mean(axis=0)*100), dtype=np.float64)
C7AIR_comp2_SD_RF = np.std(list(C7AIR_comp2_ranger.mean(axis=0)*100), dtype=np.float64)
# print("{:.15f}".format(C7AIR_raw_SD_RF))
C7AIR_SD_RF = [C7AIR_raw_SD_RF, C7AIR_comp1_SD_RF, C7AIR_comp2_SD_RF]
print(C7AIR_SD_RF)

compression_level = ['0%', '57%', '94%']
accuracy_data_ranger = [C7AIR_raw_mean_ranger, C7AIR_comp1_mean_ranger, C7AIR_comp2_mean_ranger]
accuracy_df_ranger = pd.DataFrame(np.column_stack(accuracy_data_ranger))
accuracy_df_ranger.columns = list(compression_level)
accuracy_df_ranger = accuracy_df_ranger.transpose()
accuracy_df_ranger.columns = ['Plant Height']
accuracy_df_ranger = np.array(accuracy_df_ranger)
# print(accuracy_df_ranger)
#accuracy_df_ranger = accuracy_df_ranger/accuracy_df_ranger[0]*100
accuracy_df_ranger = pd.DataFrame(accuracy_df_ranger, columns=['RF'], index=compression_level)
print(accuracy_df_ranger)

# processing GBLUP Prediction Accuracy
C7AIR_raw_gblup = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                       'C7AIR_PredictionAccuracy_Raw_GBLUP.csv', low_memory=False)
C7AIR_comp1_gblup = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_1_GBLUP.csv', low_memory=False)
C7AIR_comp2_gblup = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_2_GBLUP.csv', low_memory=False)

# set height of bar
C7AIR_raw_mean_gblup = statistics.mean(list(C7AIR_raw_gblup.mean(axis=0)))
C7AIR_comp1_mean_gblup= statistics.mean(list(C7AIR_comp1_gblup.mean(axis=0)))
C7AIR_comp2_mean_gblup = statistics.mean(list(C7AIR_comp2_gblup.mean(axis=0)))
# print(C7AIR_raw_mean_gblup)
## Calculate Standard Deviation
C7AIR_raw_SD_gblup = np.std(list(C7AIR_raw_gblup.mean(axis=0)*100), dtype=np.float64)
C7AIR_comp1_SD_gblup = np.std(list(C7AIR_comp1_gblup.mean(axis=0)*100), dtype=np.float64)
C7AIR_comp2_SD_gblup = np.std(list(C7AIR_comp2_gblup.mean(axis=0)*100), dtype=np.float64)

C7AIR_SD_gblup = [C7AIR_raw_SD_gblup, C7AIR_comp1_SD_gblup, C7AIR_comp2_SD_gblup]
print(C7AIR_SD_gblup)

accuracy_data_gblup = [C7AIR_raw_mean_gblup, C7AIR_comp1_mean_gblup, C7AIR_comp2_mean_gblup]
accuracy_df_gblup = pd.DataFrame(np.column_stack(accuracy_data_gblup))
accuracy_df_gblup = accuracy_df_gblup.transpose()
accuracy_df_gblup = np.array(accuracy_df_gblup)
# print(accuracy_df_gblup)
#accuracy_df_gblup = accuracy_df_gblup/accuracy_df_gblup[0]*100
accuracy_df_gblup = pd.DataFrame(accuracy_df_gblup, columns=['GBLUP'], index=compression_level)
print(accuracy_df_gblup)

# processing BayesB Prediction Accuracy
C7AIR_raw_BayesB = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                       'C7AIR_PredictionAccuracy_Raw_BayesB.csv', low_memory=False)
C7AIR_comp1_BayesB = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_1_BayesB.csv', low_memory=False)
C7AIR_comp2_BayesB = pd.read_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Compress_2_BayesB.csv', low_memory=False)

# set height of bar
C7AIR_raw_mean_BayesB = statistics.mean(list(C7AIR_raw_BayesB.mean(axis=0)))
C7AIR_comp1_mean_BayesB = statistics.mean(list(C7AIR_comp1_BayesB.mean(axis=0)))
C7AIR_comp2_mean_BayesB = statistics.mean(list(C7AIR_comp2_BayesB.mean(axis=0)))

## Calculate Standard Deviation
C7AIR_raw_SD_BayesB = np.std(list(C7AIR_raw_BayesB.mean(axis=0)*100), dtype=np.float64)
C7AIR_comp1_SD_BayesB = np.std(list(C7AIR_comp1_BayesB.mean(axis=0)*100), dtype=np.float64)
C7AIR_comp2_SD_BayesB = np.std(list(C7AIR_comp2_BayesB.mean(axis=0)*100), dtype=np.float64)

C7AIR_SD_BayesB = [C7AIR_raw_SD_BayesB, C7AIR_comp1_SD_BayesB, C7AIR_comp2_SD_BayesB]
print(C7AIR_SD_BayesB)

accuracy_data_BayesB = [C7AIR_raw_mean_BayesB, C7AIR_comp1_mean_BayesB, C7AIR_comp2_mean_BayesB]
accuracy_df_BayesB = pd.DataFrame(np.column_stack(accuracy_data_BayesB))
accuracy_df_BayesB = accuracy_df_BayesB.transpose()
accuracy_df_BayesB = np.array(accuracy_df_BayesB)
# print(accuracy_df_BayesB)
#accuracy_df_BayesB = accuracy_df_BayesB/accuracy_df_BayesB[0]*100
accuracy_df_BayesB = pd.DataFrame(accuracy_df_BayesB, columns=['BayesB'], index=compression_level)
# print(accuracy_df_BayesB)

accuracy_df = pd.concat([accuracy_df_BayesB, accuracy_df_gblup, accuracy_df_ranger], axis=1, sort=False)
print(accuracy_df)

# accuracy_df.to_csv('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
#                         'C7AIR_PredictionAccuracy_RF_GBLUP_BayesB.csv')

ax = accuracy_df.plot(kind="bar", figsize=(16, 12), rot=0, capsize=6,
                 yerr=[C7AIR_SD_BayesB, C7AIR_SD_gblup, C7AIR_SD_RF])
ax = accuracy_df.plot(kind="bar", figsize=(16, 12), rot=0, capsize=6)
plt.title("Prediction Accuracy of BayesB, GBLUP and RF (C7AIR)", fontsize=24)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#print(accuracy_df.index)
for p in ax.patches:
    ax.annotate(str("{:.2f}".format(p.get_height())), (p.get_x() + p.get_width() / 2,
                    p.get_height()), fontsize=15, ha='center', va='center',
                xytext=(0, 8), textcoords='offset points')
# # plt.savefig('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
# #                        'C7AIR_PredictionAccuracy.png')
plt.savefig('../C7AIR_Data/C7AIR_Prediction_Accuracy/'
                         'C7AIR_PredictionAccuracy_Ranger_GBLUP_BayesB.png')
plt.show()
