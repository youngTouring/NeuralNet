from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from itertools import chain
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import random
import os

def LoadDataFromAnn():
    peaks_y_values_collect = []
    prominence_collect = []
    width_collect = []
    peaks_y_values_minima_collect = []
    labels_collect = []
    files = []
    path = 'Dataset'
    for file in os.listdir(path):
        files.append(file)
    # shuffling files for more randomly uniformed dataset
    random.shuffle(files)
    for f in files:
        with open(f'{path}/{f}', 'rb') as file:
            data = pickle.load(file)
            peaks_y_values = data[0]
            amplitude = data[1]
            latitude = data[2]
            peaks_y_values_minima = data[3]
            peaks_labels = data[4]
            # appending data from all files to certain lists
            peaks_y_values_collect.append(peaks_y_values)
            prominence_collect.append(amplitude)
            width_collect.append(latitude)
            peaks_y_values_minima_collect.append(peaks_y_values_minima)
            labels_collect.append(peaks_labels)
    # dimension reduction
    peaks_y_values_collect = list(chain.from_iterable(peaks_y_values_collect))
    prominence_collect = list(chain.from_iterable(prominence_collect))
    width_collect = list(chain.from_iterable(width_collect))
    peaks_y_values_minima_collect = list(chain.from_iterable(peaks_y_values_minima_collect))
    labels_collect = list(chain.from_iterable(labels_collect))
    # dataframe creation
    data_dict = {'Peaks': peaks_y_values_collect, 'Amplitude': prominence_collect,
     'Width': width_collect,'Minima': peaks_y_values_minima_collect, 'Labels': labels_collect}
    return data_dict

def AnnDataFrame():
    data = LoadDataFromAnn()
    signal_dataframe = pd.DataFrame(data=data)
    # Numeric values and labels extraction
    data,labels = signal_dataframe.iloc[:,:4],signal_dataframe['Labels']
    labels = np.ravel(labels)
    # spliting data into training/testing features/labels
    RANDOM_SEED = 40
    tf.random.set_seed(RANDOM_SEED)
    train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.2,random_state=RANDOM_SEED,shuffle=True)
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    # determine the number of input features
    n_features = train_data.shape[1]
    return train_data,test_data,train_labels,test_labels, n_features

def TrainNetwork():
    train_data,test_data,train_labels,test_labels,input_shape = AnnDataFrame()
    steps_per_epoch = np.ceil(len(train_data) / 32)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(64,activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(128,activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    # Podsumowanie struktury modelu:
    model.summary()
    # Kompilacja modelu sieci:
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    # Trening sieci:
    history = model.fit(train_data, train_labels, epochs=8, batch_size=32,validation_data=(test_data,test_labels),
                        steps_per_epoch=steps_per_epoch)
    model.save('peaks_classifier_model.h5')
    # Wykres z dokładnością treningu oraz funckją straty:
    hist = history.history
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(hist['loss'], lw=3)
    ax.set_title('Funkcja straty uczenia', size=15)
    ax.set_xlabel('Epoka', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(hist['accuracy'], lw=3)
    ax.set_title('Dokładność treningu', size=15)
    ax.set_xlabel('Epoka', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig('krzywa_uczenia.pdf')
    plt.show()
    return test_data, test_labels

def Predicition():
    test,lab = TrainNetwork()
    new_model = tf.keras.models.load_model('peaks_classifier_model_.h5')
    prediction = new_model.predict_classes(test)
    fails = []
    for i in range(len(test)):
        # print(f'{test[i].tolist()} => {prediction[i]} (expected {lab[i]})')
        if prediction[i] != lab[i]:
            fails.append(prediction[i])
    print(fails)
    accuracy = len(fails)/len(test) * 100
    print(accuracy)
Predicition()
