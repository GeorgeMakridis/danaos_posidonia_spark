import itertools
import time

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from demo_1 import Configs


def clip_data(data_in=None):
    for column in data_in.columns:
        if column in list(Configs.CLIP_DICT.keys()):
            data_in[column] = data_in[column].clip(lower=Configs.CLIP_DICT[column][0],
                                                   upper=Configs.CLIP_DICT[column][1])
    return data_in


def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def clean_data(data_in):
    data = data_in.fillna(method='ffill')
    data = data.fillna(method='bfill')
    return data


def get_split_prep_data(data, train_start, train_end,
                        column_name):
    data = clean_data(data)

    minmax = MinMaxScaler(feature_range=(-1, 1))

    data[column_name] = minmax.fit_transform(data[column_name].values.reshape(-1, 1))

    data = data[column_name]

    result = []

    for index in range(train_start, train_end - Configs.LSTM_SEQUENCE_LENGTH):
        result.append(data[index: index + Configs.LSTM_SEQUENCE_LENGTH])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    train = result[train_start:train_end, :]
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train,


def prepare_data(data=None, column_name=None):
    x_train, y_train = get_split_prep_data(data, 0, len(data), column_name)

    return x_train, y_train


def build_model(num_features):
    model = Sequential()
    layers = {'input': num_features, 'hidden1': 200, 'hidden2': 500, 'hidden3': 210, 'output': num_features}

    model.add(LSTM(
        input_length=Configs.LSTM_SEQUENCE_LENGTH - 1,
        input_dim=layers['input'],
        output_dim=layers['hidden1'],
        return_sequences=True, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(LSTM(
        layers['hidden2'],
        return_sequences=True, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(LSTM(
        layers['hidden2'],
        return_sequences=True, activation='tanh'))
    model.add(Dropout(0.5))
    #
    model.add(LSTM(
        layers['hidden2'],
        return_sequences=True, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(LSTM(
        layers['hidden3'],
        return_sequences=False, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(
        output_dim=layers['output'], activation='linear'))

    start = time.time()
    model.compile(loss="mape", optimizer="adam", )
    print("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None, column_name=None, X_train=None, y_train=None, X_test=None, y_test=None,
                alpha=None, num_features=0):
    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model(num_features)

    print("Training...")
    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5,verbose=0, mode='auto', )
    # tensorboard = TensorBoard("logs/unsupervised_dnn_approach")
    callbacks_list = [earlystop]
    model.fit(X_train, y_train,
              batch_size=Configs.LSTM_BATCH_SIZE, nb_epoch=Configs.LSTM_EPOCHS, callbacks=callbacks_list, verbose=1,
              validation_split=0.3)

    model.save('unsupervised_dnn_with_datction_approach_diff_' + str(alpha) + '.h5')

    return model


def s_entropy(freq_list):
    ''' This function computes the shannon entropy of a given frequency distribution.
    USAGE: shannon_entropy(freq_list)
    ARGS: freq_list = Numeric vector represnting the frequency distribution
    OUTPUT: A numeric value representing shannon's entropy'''
    freq_list = [element for element in freq_list if element != 0]
    sh_entropy = 0.0
    for freq in freq_list:
        sh_entropy += freq * np.log(freq)
    sh_entropy = -sh_entropy
    return (sh_entropy)


def p_entropy(ts, embdim=3, embdelay=1):
    ordinal_pat = weighted_ordinal_patterns(ts, embdim, embdelay)
    max_entropy = np.log(len(ordinal_pat))
    p = np.divide(np.array(ordinal_pat), float(sum(ordinal_pat)))
    return (s_entropy(p) / max_entropy)


def ordinal_patterns(ts, embdim, embdelay):
    """ This function computes the ordinal patterns of a time series for a given embedding dimension and embedding delay.
    USAGE: ordinal_patterns(ts, embdim, embdelay)
    ARGS: ts = Numeric vector represnting the time series, embdim = embedding dimension (3<=embdim<=7 prefered range), embdelay =  embdding delay
    OUPTUT: A numeric vector representing frequencies of ordinal patterns"""
    time_series = ts
    possible_permutations = list(itertools.permutations(range(embdim)))
    lst = list()
    for i in range(len(time_series) - embdelay * (embdim - 1)):
        sorted_index_array = list(np.argsort(time_series[i:(embdim + i)]))
        lst.append(sorted_index_array)
    lst = np.array(lst)
    element, freq = np.unique(lst, return_counts=True, axis=0)
    freq = list(freq)
    if len(freq) != len(possible_permutations):
        for i in range(len(possible_permutations) - len(freq)):
            freq.append(0)
        return (freq)
    else:
        return (freq)


def weighted_ordinal_patterns(ts, embdim, embdelay):
    time_series = ts
    possible_permutations = list(itertools.permutations(range(embdim)))
    temp_list = list()
    wop = list()
    for i in range(len(time_series) - embdelay * (embdim - 1)):
        Xi = time_series[i:(embdim + i)]
        Xn = time_series[(i + embdim - 1): (i + embdim + embdim - 1)]
        Xi_mean = np.mean(Xi)
        Xi_var = (Xi - Xi_mean) ** 2
        weight = np.mean(Xi_var)
        sorted_index_array = list(np.argsort(Xi))
        temp_list.append([''.join(map(str, sorted_index_array)), weight])
    result = pd.DataFrame(temp_list, columns=['pattern', 'weights'])
    freqlst = dict(result['pattern'].value_counts())
    for pat in (result['pattern'].unique()):
        wop.append(np.sum(result.loc[result['pattern'] == pat, 'weights'].values))
    return (wop)


def get_mean_of_cyl_values(dataframe):
    scavAirFireDetTemp_cols = [col for col in dataframe.columns if col.startswith('scavAirFireDetTempN')]
    cylExhGasOutTemp_cols = [col for col in dataframe.columns if col.startswith('cylExhGasOutTempN')]
    cylJCFWOutTemp_cols = [col for col in dataframe.columns if col.startswith('cylJCFWOutTempNo')]
    cylPistonCOOutTemp_cols = [col for col in dataframe.columns if col.startswith('cylPistonCOOutTempNo')]
    tcExhGasInTemp_cols = [col for col in dataframe.columns if col.startswith('tcExhGasInTempN')]
    tcExhGasOutTemp_cols = [col for col in dataframe.columns if col.startswith('tcExhGasOutTempN')]
    tcLOInLETPress_cols = [col for col in dataframe.columns if col.startswith('tcLOInLETPressNo')]
    tcLOOutLETTemp_cols = [col for col in dataframe.columns if col.startswith('tcLOOutLETTempNo')]
    coolingWOutLETTemp_cols = [col for col in dataframe.columns if col.startswith('coolingWOutLETTemp')]
    tcRPM_cols = [col for col in dataframe.columns if col.startswith('tcRPM')]

    dataframe['scavAirFireDetTempN'] = dataframe[scavAirFireDetTemp_cols].mean(axis=1)
    dataframe['cylExhGasOutTempN'] = dataframe[cylExhGasOutTemp_cols].mean(axis=1)
    dataframe['cylJCFWOutTempNo'] = dataframe[cylJCFWOutTemp_cols].mean(axis=1)
    dataframe['cylPistonCOOutTempNo'] = dataframe[cylPistonCOOutTemp_cols].mean(axis=1)
    dataframe['tcExhGasInTempN'] = dataframe[tcExhGasInTemp_cols].mean(axis=1)
    dataframe['tcExhGasOutTempN'] = dataframe[tcExhGasOutTemp_cols].mean(axis=1)
    dataframe['tcLOInLETPressNo'] = dataframe[tcLOInLETPress_cols].mean(axis=1)
    dataframe['tcLOOutLETTempNo'] = dataframe[tcLOOutLETTemp_cols].mean(axis=1)
    dataframe['coolingWOutLETTemp'] = dataframe[coolingWOutLETTemp_cols].max(axis=1)
    dataframe['tcRPM'] = dataframe[tcRPM_cols].mean(axis=1)

    dataframe.drop(scavAirFireDetTemp_cols, axis=1, inplace=True)
    dataframe.drop(cylExhGasOutTemp_cols, axis=1, inplace=True)
    dataframe.drop(cylJCFWOutTemp_cols, axis=1, inplace=True)
    dataframe.drop(cylPistonCOOutTemp_cols, axis=1, inplace=True)
    dataframe.drop(tcExhGasInTemp_cols, axis=1, inplace=True)
    dataframe.drop(tcExhGasOutTemp_cols, axis=1, inplace=True)
    dataframe.drop(tcLOInLETPress_cols, axis=1, inplace=True)
    dataframe.drop(tcLOOutLETTemp_cols, axis=1, inplace=True)
    dataframe.drop(coolingWOutLETTemp_cols, axis=1, inplace=True)
    dataframe.drop(tcRPM_cols, axis=1, inplace=True)
    return dataframe

def clean_svm(vds_5_in):
    vds_5 = vds_5_in.fillna(method='ffill')
    vds_5 = vds_5.fillna(method='bfill')

    minmax = MinMaxScaler()

    vds_5 = minmax.fit_transform(vds_5)

    from joblib import dump
    dump(minmax, 'minmax.joblib')

    return vds_5