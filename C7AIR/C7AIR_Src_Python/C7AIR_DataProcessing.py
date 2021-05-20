# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 06.10.2020
# =========================================

from keras import Input
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# loading data
def SevenkArray_GenData():
    df = pd.read_csv('../C7AIR_Data/C7AIR_Raw/C7AIR_Genotype.csv', low_memory=False)
    df_transpose = df.transpose()
    df_range = df_transpose.iloc[1:, 0:]
    gen_data = np.array(df_range)
    return gen_data.T


print(SevenkArray_GenData())
print('Shape of data before pre-processing: ', SevenkArray_GenData().shape)


def OneHotEncode():
    X = SevenkArray_GenData()
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
onehotinp = np.hsplit(OneHotEncode(), 1014)
split_input = np.array(onehotinp)
print(split_input[0])
print("Splitting Input data: ", split_input[0].shape)

index = 0
for i in range(len(split_input)):
    # print('Splitting input layer data: ', index, split_input[i].shape)
    # deep autoencoder
    input = Input(shape=(split_input[i].shape[1],))
    print('Shape of input layer data: ', index, input.shape)
    np.savetxt('../C7AIR_Data/C7AIR_OneHotEncode/C7AIR_OneHot_{}'.format(index)
               + '.csv', split_input[i], delimiter=",", fmt='%1.0f')
    index += 1