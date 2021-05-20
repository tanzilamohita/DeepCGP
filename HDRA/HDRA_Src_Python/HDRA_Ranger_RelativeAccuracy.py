# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 3/29/2021
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hdra_accuracy = pd.read_csv('../HDRA_Data/HDRA_Prediction_Accuracy/HDRA_PredictionAccuracy.csv',
                            low_memory=False).iloc[0:4, 1:-1]
hdra_accuracy = hdra_accuracy.transpose()
# print(hdra_accuracy)
# hdra_accuracy.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
#                    'HDRA_Prediction_Accuracy_T.csv', float_format='%1.3f',
#                      header=('0%', '57%', '93%', '98%'))

rel_acc_all = []
traits_all = []
for traits, accuracy in hdra_accuracy.iterrows():
    #print(traits)
    traits_all.append(traits)
    rel_acc_apnd = list()
    for comp_lev in range(len(accuracy)):
        rel_acc = accuracy[comp_lev]/accuracy[0]*100
        #print(rel_acc)
        rel_acc_apnd.append(rel_acc)
    #print("----------------")
    #print(rel_acc_apnd)
    rel_acc_all.append(rel_acc_apnd)

rel_acc_all = pd.DataFrame(rel_acc_all, columns=('0%', '57%', '93%', '98%'), index=traits_all)
print(rel_acc_all)
rel_acc_all.to_csv('../HDRA_Data/HDRA_Prediction_Accuracy/'
                   'HDRA_Prediction_RelativeAccuracy.csv', float_format='%1.3f')









