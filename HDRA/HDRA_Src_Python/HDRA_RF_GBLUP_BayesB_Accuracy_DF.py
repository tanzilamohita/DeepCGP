# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 6/30/2021
# ===============================
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt

# processing Ranger Prediction Accuracy
HDRA_raw_ranger = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                       'HDRA_PredictionAccuracyProcessed_Raw.csv', low_memory=False)
HDRA_comp1_ranger = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_1.csv', low_memory=False)
HDRA_comp2_ranger = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_2.csv', low_memory=False)
HDRA_comp3_ranger = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_3.csv', low_memory=False)

# set height of bar
HDRA_raw_mean_ranger = list(HDRA_raw_ranger.mean(axis=0)*100)
HDRA_comp1_mean_ranger= list(HDRA_comp1_ranger.mean(axis=0)*100)
HDRA_comp2_mean_ranger = list(HDRA_comp2_ranger.mean(axis=0)*100)
HDRA_comp3_mean_ranger = list(HDRA_comp3_ranger.mean(axis=0)*100)

HDRA_Traits = list(HDRA_raw_ranger.columns)
del HDRA_Traits[0]
HDRA_Traits = list(map(lambda st: str.replace(st, ".", " "), HDRA_Traits))
# print(HDRA_Traits)

compression_level_Ranger = ['0%', '57%', '93%', '98%']
accuracy_data_ranger = [HDRA_raw_mean_ranger, HDRA_comp1_mean_ranger, HDRA_comp2_mean_ranger,
                        HDRA_comp3_mean_ranger]
accuracy_df_ranger = pd.DataFrame(np.column_stack(accuracy_data_ranger))
accuracy_df_ranger.columns = list(compression_level_Ranger)
# accuracy_df_ranger = accuracy_df_ranger.transpose()
accuracy_df_ranger = np.array(accuracy_df_ranger)
#print(accuracy_df_ranger[0])
#accuracy_df_ranger = accuracy_df_ranger/accuracy_df_ranger[0]*100
accuracy_df_ranger = pd.DataFrame(accuracy_df_ranger, index=HDRA_Traits,
                                  columns=compression_level_Ranger)
# print(accuracy_df_ranger)
# accuracy_df_ranger.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                         'HDRA_PredictionAccuracy_RF_DF.csv', float_format='%1.2f')

# processing GBLUP Prediction Accuracy
# HDRA_raw_gblup = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                        'HDRA_PredictionAccuracy_Raw_GBLUP.csv', low_memory=False)
HDRA_comp1_gblup = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_1_GBLUP.csv', low_memory=False)
HDRA_comp2_gblup = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_2_GBLUP.csv', low_memory=False)
HDRA_comp3_gblup = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_3_GBLUP.csv', low_memory=False)

# set height of bar
# HDRA_raw_mean_gblup = list(HDRA_raw_gblup.mean(axis=0)*100)
HDRA_comp1_mean_gblup= list(HDRA_comp1_gblup.mean(axis=0)*100)
HDRA_comp2_mean_gblup = list(HDRA_comp2_gblup.mean(axis=0)*100)
HDRA_comp3_mean_gblup = list(HDRA_comp3_gblup.mean(axis=0)*100)

compression_level_gblup = ['57%', '93%', '98%']
accuracy_data_gblup = [HDRA_comp1_mean_gblup, HDRA_comp2_mean_gblup,
                       HDRA_comp3_mean_gblup]
accuracy_df_gblup = pd.DataFrame(np.column_stack(accuracy_data_gblup))
# accuracy_df_gblup = accuracy_df_gblup.transpose()
accuracy_df_gblup = np.array(accuracy_df_gblup)
# print(accuracy_df_gblup)
#accuracy_df_gblup = accuracy_df_gblup/accuracy_df_gblup[0]*100
accuracy_df_gblup = pd.DataFrame(accuracy_df_gblup, columns=compression_level_gblup,
                                 index=HDRA_Traits)
# print(accuracy_df_gblup)
# accuracy_df_gblup.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                         'HDRA_PredictionAccuracy_GBLUP_DF.csv', float_format='%1.2f')
#
# processing BayesB Prediction Accuracy
# HDRA_raw_BayesB = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                        'HDRA_PredictionAccuracy_Raw_BayesB.csv', low_memory=False)
HDRA_comp1_BayesB = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_1_BayesB.csv', low_memory=False)
HDRA_comp2_BayesB = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_2_BayesB.csv', low_memory=False)
HDRA_comp3_BayesB = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                         'HDRA_PredictionAccuracyProcessed_Compress_3_BayesB.csv', low_memory=False)

# set height of bar
# HDRA_raw_mean_BayesB = list(HDRA_raw_BayesB.mean(axis=0)*100)
HDRA_comp1_mean_BayesB = list(HDRA_comp1_BayesB.mean(axis=0)*100)
HDRA_comp2_mean_BayesB = (HDRA_comp2_BayesB.mean(axis=0)*100)
HDRA_comp3_mean_BayesB = list(HDRA_comp3_BayesB.mean(axis=0)*100)
# print(list(HDRA_comp3_BayesB.mean(axis=0)*100))

compression_level_BayesB = ['57%', '93%', '98%']
accuracy_data_BayesB = [HDRA_comp1_mean_BayesB, HDRA_comp2_mean_BayesB,
                        HDRA_comp3_mean_BayesB]
accuracy_df_BayesB = pd.DataFrame(np.column_stack(accuracy_data_BayesB))
# accuracy_df_BayesB = accuracy_df_BayesB.transpose()
accuracy_df_BayesB = np.array(accuracy_df_BayesB)
# print(accuracy_df_BayesB)
#accuracy_df_BayesB = accuracy_df_BayesB/accuracy_df_BayesB[0]*100
accuracy_df_BayesB = pd.DataFrame(accuracy_df_BayesB, columns=compression_level_BayesB,
                                  index=HDRA_Traits)
# print(accuracy_df_BayesB)
# accuracy_df_BayesB.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                         'HDRA_PredictionAccuracy_BayesB_DF.csv', float_format='%1.2f')

# accuracy_df = pd.concat([accuracy_df_BayesB, accuracy_df_gblup, accuracy_df_ranger],
#                         axis=1, sort=False)
# print(accuracy_df)
#Prediction_Methods = ['BayesB', 'GBLUP', 'RF']
arrays_comp1 = [['57%', '57%', '57%'], ['BayesB', 'GBLUP', 'RF']]
tuples_comp1 = list(zip(*arrays_comp1))
index_comp1 = pd.MultiIndex.from_tuples(tuples_comp1)

HDRA_comp1 = [HDRA_comp1_mean_BayesB, HDRA_comp1_mean_gblup, HDRA_comp1_mean_ranger]
HDRA_comp1 = pd.DataFrame(np.column_stack(HDRA_comp1))
HDRA_comp1 = np.array(HDRA_comp1)
HDRA_comp1 = pd.DataFrame(HDRA_comp1, columns=index_comp1,
                                  index=HDRA_Traits)
#print(HDRA_comp1)

arrays_comp2 = [['93%', '93%', '93%'], ['BayesB', 'GBLUP', 'RF']]
tuples_comp2 = list(zip(*arrays_comp2))
index_comp2 = pd.MultiIndex.from_tuples(tuples_comp2)

HDRA_comp2 = [HDRA_comp2_mean_BayesB, HDRA_comp2_mean_gblup, HDRA_comp2_mean_ranger]
HDRA_comp2 = pd.DataFrame(np.column_stack(HDRA_comp2))
HDRA_comp2 = np.array(HDRA_comp2)
HDRA_comp2 = pd.DataFrame(HDRA_comp2, columns=index_comp2,
                                  index=HDRA_Traits)
#print(HDRA_comp2)

arrays_comp3 = [['98%', '98%', '98%'], ['BayesB', 'GBLUP', 'RF']]
tuples_comp3 = list(zip(*arrays_comp3))
index_comp3 = pd.MultiIndex.from_tuples(tuples_comp3)

HDRA_comp3 = [HDRA_comp3_mean_BayesB, HDRA_comp3_mean_gblup, HDRA_comp3_mean_ranger]
HDRA_comp3 = pd.DataFrame(np.column_stack(HDRA_comp3))
HDRA_comp3 = np.array(HDRA_comp3)
HDRA_comp3 = pd.DataFrame(HDRA_comp3, columns=index_comp3,
                                  index=HDRA_Traits)

#print(HDRA_comp3)

HDRA_DF = pd.concat([HDRA_comp1, HDRA_comp2, HDRA_comp3],
                        axis=1, sort=False)
print(HDRA_DF)
# HDRA_DF.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                         'HDRA_BayesB_GBLUP_RF.csv', float_format='%1.2f')