# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 21.10.2020
# =========================================

from sklearn.model_selection import train_test_split
from keras import Input, Model
from keras.layers import Dense
import numpy as np
import os
import keras
import glob
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error
import time
import pathlib
from numpy import mean

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
# using GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # model will be trained on GPU 0, if it is -1, GPU will not use for training

# load Processed Data
DigitalInput_fileList = natsorted(glob.glob("../HDRA_Data/HDRA_OneHot/"+"*.csv"))
DigitalInput_fileList = DigitalInput_fileList[0:1]
# [0:1] -> It will run 1st csv file;
# If you want to run 10000 files at a time, set [0:10000]
print(DigitalInput_fileList)

start_time = time.time()
evaluate = []
mse_loss = []
index = 0
for DigitalInput_filename in DigitalInput_fileList:
    digital_data = pd.read_csv(DigitalInput_filename, low_memory=False, header=None)
    digital_data = np.array(digital_data)
    # print(index, digital_data)

    # Creating a training set, test set and validation set
    x_train, x_mid = train_test_split(digital_data, test_size=0.4)
    x_test, x_valid = train_test_split(x_mid, test_size=0.5)
    print('Shape of train data: ', x_train.shape)
    print('Shape of test data: ', x_test.shape)
    print('Shape of validation data: ', x_valid.shape)

    # Taking the input data of dimension 28 and convert it to keras tensors.
    input = Input(shape=(digital_data.shape[1],))
    print('Shape of input layer data: ', index, input.shape)

    # For all the hidden layers for the encoder and decoder
    # we use relu activation function for non-linearity.
    encoded = Dense(14, activation='relu')(input)
    encoded = Dense(7, activation='relu')(encoded)
    encoded = Dense(3, activation='sigmoid')(encoded)

    decoded = Dense(7, activation='relu')(encoded)
    decoded = Dense(14, activation='relu')(decoded)
    # The output layer needs to predict the probability of an output
    # which needs to either 0 or 1 and hence we use sigmoid activation function.
    decoded = Dense(digital_data.shape[1], activation='sigmoid')(decoded)

    # this model maps an input to its reconstruction
    # creating the autoencoder with input data. Output will be the final decoder layer.
    autoencoder = Model(input, decoded)
    # extracting the encoder which takes input data and the output of encoder
    # encoded data of dimension 3
    encoder = Model(input, encoded)
    # the structure of the deep autoencoder model
    autoencoder.summary()
    # the structure of the encoder
    encoder.summary()
    # plot_model(encoder, to_file='../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_Flowgraph/'
    #                              'HDRA_Net_1_EncFlowgraph_{}'.format(index) + '.png', show_shapes=True)
    # plot_model(autoencoder, to_file='../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_Flowgraph/'
    #                              'HDRA_Net_1_Flowgraph_{}'.format(index) + '.png', show_shapes=True)
    # # compiling the autoencoder model with adam optimizer.
    adm = keras.optimizers.Adam(lr=0.001)
    # We use MSE to calculate the loss of the model
    autoencoder.compile(optimizer=adm, loss='mse')
    # We finally train the autoencoder using the training data with 100 epochs and batch size of 52.
    history = autoencoder.fit(x_train, x_train,
                              epochs=200,
                              batch_size=52,
                              shuffle=True,
                              validation_data=(x_valid, x_valid))

    # predicting the test using encoder
    enc_out = encoder.predict(digital_data)
    # save encoded data into CSV file
    np.savetxt('../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_EncData/'
               'HDRA_Net_1_EncData_{}'.format(index) + '.csv', enc_out, delimiter=",", fmt='%1.0f')

    # reconstructing the input from autoencoder
    dec_out = autoencoder.predict(digital_data)
    print('Shape of Decoded Data Output: ', dec_out.shape)
    print(dec_out)

    # save decoded data into CSV file
    np.savetxt('../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_DecData/HDRA_Net_1_DecData_{}'.format(index)
               + '.csv', dec_out, delimiter=",", fmt='%1.0f')


    # save model
    filename = '../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_Model/HDRA_Net_1_Model_{}'.format(index) + '.h5'
    autoencoder.save(filename)
    score = autoencoder.evaluate(x_test, x_test, verbose=0)
    score_loss = score * 100

    # score_array = np.column_stack((score_loss, score_acc))
    evaluate.append([score_loss])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    result = np.array([loss, val_loss])

    print(history.history.keys())
    print('Loss: {:.3f}'.format(score_loss))
    mse = mean_squared_error(digital_data, dec_out)
    print('SKlearn MSE: ', mse)
    mse_loss.append([mse])

    # plotting history for loss
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_Plot/HDRA_Net_1_PlotLoss_{}'.format(index) + '.png')
    # plt.show()
    plt.clf()
    index += 1

np.savetxt('../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_Evaluate/'
        'HDRA_Net_1_Evaluate.csv',
        evaluate, delimiter=',', fmt='%1.3f', comments='', header='Loss')
np.savetxt('../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_MSE/'
           'HDRA_Net_1_MSE.csv',
           mse_loss, delimiter=',', fmt='%1.5f', comments='', header='MSE Loss')

pathlib.Path("../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_Time/"
        "HDRA_Net_1_TrainingTime.txt")\
        .write_text("HDRA_Net_1_Training_Time: {}"
        .format(time.time() - start_time))
pathlib.Path("../HDRA_Data/HDRA_ModelMetaData/HDRA_Net_1/HDRA_Net_1_MSE/"
        "HDRA_Net_1_MSE.txt").write_text("HDRA Net_1 Mean MSE Loss: {}"
        .format(mean(mse_loss)))

print("MSE Loss: ", mean(mse_loss))
print('Total Training Time: ', time.time() - start_time)

