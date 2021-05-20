# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 20.10.2020
# =========================================

import numpy as np
import pandas as pd
from keras import Input
import os
import time

# using GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # model will be trained on GPU 0, if it is -1, GPU will not use for training

start_time = time.time()
print("Loading Data")
# loading data
def HDRA_GenData():
    df = pd.read_csv('../HDRA_Data/HDRA_Raw/HDRA-G6-4-RDP1-RDP2-NIAS.AGCT.csv', low_memory=False, na_values=["N"])
    df_transpose = df.transpose()
    df_range = df_transpose.iloc[1:, 0:]
    gen_data = np.array(df_range)
    return gen_data.T


print("The shape of data: ", HDRA_GenData().shape)


# print(HDRA_GenData())
# missing_values = HDRA_GenData().isnull().sum().sum()
# print("Total Missing values:", missing_values)

def OneHotEncode():
    X = HDRA_GenData()
    onehotlabels = []
    for i in range(len(X)):
        # print(X[i])
        item = X[i]
        onehot = []
        for j in range(len(item)):
            # element = item[j]
            if item[j] == 'A':
                onehot.extend((1, 0, 0, 0))
            elif item[j] == 'C':
                onehot.extend((0, 1, 0, 0))
            elif item[j] == 'G':
                onehot.extend((0, 0, 1, 0))
            elif item[j] == 'T':
                onehot.extend((0, 0, 0, 1))
            else:
                onehot.extend((0, 0, 0, 0))

        onehotlabels.append(onehot)
    return np.array(onehotlabels)


print('Shape of data after pre-processing: ', OneHotEncode().shape)

# splitting input data
onehotinp = np.hsplit(OneHotEncode(), 100000)
split_input = np.array(onehotinp)
print(split_input[0])
print("Splitting Input data: ", split_input[0].shape)

index = 0
for i in range(len(split_input)):
    # print('Splitting input layer data: ', index, split_input[i].shape)
    # deep autoencoder
    input = Input(shape=(split_input[i].shape[1],))
    print('Shape of input layer data: ', index, input.shape)
    np.savetxt('../HDRA_Data/HDRA_Zero_OneHot/HDRA_Zero_OneHot_{}'.format(index)
               + '.csv', split_input[i], delimiter=",", fmt='%1.0f')
    index += 1

print((time.time() - start_time))
print("Completed")

