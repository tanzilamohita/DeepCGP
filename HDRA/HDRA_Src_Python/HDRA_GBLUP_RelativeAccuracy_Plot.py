# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 7/9/2021
# ===============================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hdra_accuracy = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/HDRA_PredictionAccuracy_GBLUP_DF.csv',
                            low_memory=False).iloc[:, 1:5]
#hdra_accuracy = hdra_accuracy.transpose()
# print(hdra_accuracy)
# hdra_accuracy.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                    'HDRA_Prediction_Accuracy_T.csv', float_format='%1.3f',
#                      header=('0%', '57%', '93%', '98%'))

rel_acc_all = []
traits_all = []
for traits, accuracy in hdra_accuracy.iterrows():
    # print(traits)
    traits_all.append(traits)
    rel_acc_apnd = list()
    for comp_lev in range(len(accuracy)):
        # print(accuracy[comp_lev])
        # print("----------------")
        # print(accuracy[1])
        rel_acc = accuracy[comp_lev]/accuracy[0] * 100
        #print(rel_acc)
        rel_acc_apnd.append(rel_acc)
    #print("----------------")
    #print(rel_acc_apnd)
    rel_acc_all.append(rel_acc_apnd)

rel_acc_all = pd.DataFrame(rel_acc_all, columns=('0%', '57%', '93%', '98%'), index=traits_all)
print(rel_acc_all)
# rel_acc_all.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                    'HDRA_Prediction_RelativeAccuracy_GBLUP.csv', float_format='%1.3f')
hdra_rel_acc = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                           'HDRA_Prediction_RelativeAccuracy_GBLUP.csv', low_memory=False, index_col=0)
hdra_rel_acc.drop(hdra_rel_acc.columns[[2]], axis=1, inplace=True)
hdra_rel_acc = hdra_rel_acc.transpose()
print(hdra_rel_acc)
hdra_rel_acc.plot.line(figsize=(15, 10))
plt.title("GBLUP Prediction Relative Accuracy for HDRA", fontsize=24)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Relative Accuracy (%)", fontsize=18)
plt.savefig('../HDRA_Data/HDRA_Prediction_Accuracy/'
                       'HDRA_Prediction_RelativeAccuracy_3 levels_GBLUP.png')
plt.show()
