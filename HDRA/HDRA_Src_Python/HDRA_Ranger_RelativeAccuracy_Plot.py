# ==============================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 3/31/2021
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hdra_rel_acc = pd.read_csv('../HDRA_Data/HDRA_Zero_Prediction_Accuracy/'
                       'HDRA_Prediction_RelativeAccuracy.csv', low_memory=False, index_col=0)
hdra_rel_acc.drop(hdra_rel_acc.columns[[2]], axis=1, inplace=True)
hdra_rel_acc = hdra_rel_acc.transpose()
print(hdra_rel_acc)
hdra_rel_acc.plot.line(figsize=(15, 10))
plt.title("Prediction Accuracy for HDRA", fontsize=24)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Compression Level", fontsize=18)
plt.ylabel("Relative Accuracy (%)", fontsize=18)
plt.savefig('../HDRA_Data/HDRA_Zero_Prediction_Accuracy/'
                       'HDRA_Prediction_RelativeAccuracy_3 levels.png')
plt.show()

