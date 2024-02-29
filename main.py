from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal

# baseline
from Baseline_Wiener.WienerFilter import *

from ESN.ESN import basicESN
from Dataset import *
from Baseline_RNN.RNNet import RNNet
import torch.utils.data as Data


def esn_prediction(myESN, train_sample, train_label, test_sample, test_label, my_signal):
    myESN.train(train_samples=train_sample,
                train_labels=train_label)

    predict_signal_vectors_train, predict_error_train = (
        myESN.predict(test_samples=train_sample, test_labels=train_label, continuous=False))

    predict_signal_vectors_test, predict_error_test = (
        myESN.predict(test_samples=test_sample, test_labels=test_label, continuous=False))

    predict_signal_train, predict_time_train, predict_signal_test, predict_time_test = (
        my_signal.signal_predict(predict_signal_vectors_train, predict_signal_vectors_test))
    return predict_signal_train, predict_time_train, predict_signal_test, predict_time_test


def rnn_prediction(myRNN, train_dataloader, test_dataloader, my_signal):
    train_loss_all, test_loss_all = myRNN.train(train_dataloader, test_dataloader)
    # myRNN.train_figure(train_loss_all, test_loss_all)
    _, predict_signal_vectors_rnn_train = myRNN.predict(train_dataloader)
    _, predict_signal_vectors_rnn_test = myRNN.predict(test_dataloader)
    predict_signal_train, predict_time_train, predict_signal_test, predict_time_test = (
        my_signal.signal_predict(predict_signal_vectors_rnn_train, predict_signal_vectors_rnn_test))
    return predict_signal_train, predict_time_train, predict_signal_test, predict_time_test


def shift_prediction(my_signal):
    predict_time = range(int(my_signal.length * my_signal.percent), my_signal.length)
    predict_signal = my_signal.signal_distorted[int(my_signal.length * my_signal.percent) - 1: -1]
    return predict_signal, predict_time


def run():
    coef = [0.95, -0.6, 0.3, 0.1]  # attenuation coefficient of AR
    # coef = [0.95]
    bias = 0
    varu = 1  # variance of active noise of AR
    varv = 0.1  # variance of AWGN
    # --------------------- signal and dataset generation ---------------------
    sample_len = 3
    label_len = 3
    node = 24  # if node is too large, ESN will be over-fit
    forcast = 1
    N = 30000  # total length
    # percent = 0.5  # N*percent points are used for training

    myESN = basicESN(size_input=sample_len,
                     size_output=label_len,
                     node=node,
                     esn_sparsity=0.04,     # 1~5 %
                     spectral_radius=0.1,   # radius larger, the capability of long memory better
                     esn_random_state=42,   # random seed number 42
                     silent=True)
    myESN.initialize_weights(None, None, None, None)

    epoch = 10
    batch = 64
    lr = 1e-3
    layer = 1
    myRNN = RNNet(0, sample_len, node, layer, sample_len,
                  epoch, batch, lr)
    myLSTM = RNNet(1, sample_len, node, layer, sample_len,
                   epoch, batch, lr)
    mywiener = wienerpredictor(coef, bias, varu, varv)

    # proportion = [2e-3 * x for x in range(1, 100, 20)]
    # proportion = [0.001]
    # proportion = [0.5]

    start_val = 5e-3        # percent
    end_val = 40
    proportion = np.logspace(np.log10(start_val/100), np.log10(end_val/100), num=8)        # from 1e-3 to 1e-1
    proportion = [0.5]

    iid_num, idd_idx = 1, 0                     # i.i.d. stochastic processes
    prediction_error = np.zeros(shape=(6, len(proportion), iid_num))

    my_signal = stochastic_signal(N, coef, bias, varu, varv, sample_len, label_len, forcast)
    for idd_idx in range(iid_num):
        my_signal.signal_generator()

        print(idd_idx)

        alpha_idx = 0
        for alpha in proportion:
            train_sample, train_label, test_sample, test_label = my_signal.dataset_generator(alpha)
            train_dataset = mydataset(train_sample, train_label,
                                      transform=transforms.Compose([ToTensor()]))
            test_dataset = mydataset(test_sample, test_label,
                                     transform=transforms.Compose([ToTensor()]))

            # --------------------- ESN ---------------------
            esn_predict_train, esn_predict_time_train, esn_predict_test, esn_predict_time_test = (
                esn_prediction(myESN, train_sample, train_label, test_sample, test_label, my_signal))

            # --------------------- some baselines ---------------------
            # Baseline: RNN
            train_dataloader = Data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
            test_dataloader = Data.DataLoader(test_dataset, batch_size=batch, shuffle=False, drop_last=True)
            rnn_predict_train, rnn_predict_time_train, rnn_predict_test, rnn_predict_time_test = (
                rnn_prediction(myRNN, train_dataloader, test_dataloader, my_signal))

            # Baseline: LSTM
            lstm_predict_train, lstm_predict_time_train, lstm_predict_test, lstm_predict_time_test = (
                rnn_prediction(myLSTM, train_dataloader, test_dataloader, my_signal))

            # Baseline: Wiener filter, know AR parameters
            '''
            wiener_predict, wiener_predict_time = mywiener.wiener_predict(my_signal.signal_distorted, N, alpha)
            order = 4
            wiener_predict_unknown, wiener_predict_unknown_time = (
                Wiener_unknown(my_signal.signal_distorted, N, alpha, order, varv))     # unknown signal
            '''

            # Baseline: directly time shift
            directshift_predict, directshift_predict_time = shift_prediction(my_signal)

            '''
            predict_figure_once(my_signal,
                                esn_predict_train, esn_predict_time_train, esn_predict_test, esn_predict_time_test,
                                rnn_predict_train, rnn_predict_time_train, rnn_predict_test, rnn_predict_time_test,
                                lstm_predict_train, lstm_predict_time_train, lstm_predict_test, lstm_predict_time_test,
                                # wiener_predict, wiener_predict_time,
                                # wiener_predict_unknown, wiener_predict_unknown_time,
                                directshift_predict, directshift_predict_time)
            '''

            # --------------------- prediction error ---------------------
            original = my_signal.signal_distorted[-len(rnn_predict_time_test):]
            prediction_error[0, alpha_idx, idd_idx] = np.sqrt(np.mean((original - esn_predict_test) ** 2))
            prediction_error[1, alpha_idx, idd_idx] = np.sqrt(np.mean((original - rnn_predict_test) ** 2))
            prediction_error[2, alpha_idx, idd_idx] = np.sqrt(np.mean((original - lstm_predict_test) ** 2))
            original = my_signal.signal_distorted[-len(directshift_predict):]
            # prediction_error[3, alpha_idx, idd_idx] = np.sqrt(np.mean((original - wiener_predict) ** 2))
            # prediction_error[4, alpha_idx, idd_idx] = np.sqrt(np.mean((original - wiener_predict_unknown) ** 2))
            prediction_error[5, alpha_idx, idd_idx] = np.sqrt(np.mean((original - directshift_predict) ** 2))
            alpha_idx += 1

    np.save("prediction_error.txt", prediction_error)

    # --------------------- result figure ---------------------
    prediction_error_mean = np.mean(prediction_error, axis=2)
    plt.figure()
    plt.plot(proportion, prediction_error_mean[0, :], marker='+', color='b', label='ESN')
    plt.plot(proportion, prediction_error_mean[1, :], marker='+', linewidth=2,
             color='r', label='RNN')
    plt.plot(proportion, prediction_error_mean[2, :], marker='+', linewidth=2,
             color='g', label='LSTM')
    '''
    plt.plot(proportion, prediction_error_mean[3, :], marker='+', linewidth=2,
             color='y', label='Wiener-known')
    plt.plot(proportion, prediction_error_mean[4, :], marker='+', linewidth=2,
             color='y', linestyle="--", label='Wiener-unknown')
    '''
    plt.plot(proportion, prediction_error_mean[5, :], marker='+', linewidth=2,
             color='c', label='Direct shift')
    plt.legend()
    plt.xlabel('Alpha')
    plt.xscale('log')
    plt.ylabel('Prediction error')
    plt.show()


def predict_figure_once(my_signal,
                        esn_predict_train, esn_predict_time_train, esn_predict_test, esn_predict_time_test,
                        rnn_predict_train, rnn_predict_time_train, rnn_predict_test, rnn_predict_time_test,
                        lstm_predict_train, lstm_predict_time_train, lstm_predict_test, lstm_predict_time_test,
                        # wiener_predict, wiener_predict_time,
                        # wiener_predict_unknown, wiener_predict_unknown_time,
                        directshift_predict, directshift_predict_time):
    plt.figure()
    plt.plot(my_signal.signal_distorted, linewidth=2,
             color='k', label='Distorted signal')
    plt.plot(my_signal.signal_original, linewidth=1,
             color='k', linestyle="--", label='Original signal')

    plt.plot(esn_predict_time_test, esn_predict_test, linewidth=1,
             color='b', label='ESN predict')
    plt.plot(esn_predict_time_train, esn_predict_train, linewidth=1,
             color='b', linestyle="--")
    plt.plot(rnn_predict_time_test, rnn_predict_test, linewidth=1,
             color='r', label='RNN predict')
    plt.plot(rnn_predict_time_train, rnn_predict_train, linewidth=1,
             color='r', linestyle="--")
    plt.plot(lstm_predict_time_test, lstm_predict_test, linewidth=1,
             color='g', label='LSTM predict')
    plt.plot(lstm_predict_time_train, lstm_predict_train, linewidth=1,
             color='g', linestyle="--")

    '''
    plt.plot(wiener_predict_time, wiener_predict, linewidth=1,
             color='y', label='Wiener predict')
    plt.plot(wiener_predict_unknown_time, wiener_predict_unknown, linewidth=1,
             color='y', linestyle="--", label='Wiener-unknown predict')
    '''
    plt.plot(directshift_predict_time, directshift_predict, linewidth=1,
             color='c', label='Direct shift predict')
    plt.legend()
    plt.xlabel('Discrete Time')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # run()

    N = 200000  # total samples for training ESN
    x_n = np.random.rand(N) * 2 - 1
    batch = 100  # ----------------- ignore at first -----------------

    '''
    b, r, eta = 0.7, 0.7*np.sqrt(2), 3/4*np.pi
    cut_length = 250
    n = np.arange(0, cut_length)
    c_n = 1 / b * r**n * np.sin(n * eta)
    h_n = (c_n - 2 * np.concatenate((np.array([0]), c_n[1:])) + 2.25 * np.concatenate((np.array([0, 0]), c_n[2:]))
           - 1.25 * np.concatenate((np.array([0, 0, 0]), c_n[3:])))
    '''

    r = 0.7 * np.sqrt(2)
    eta = np.pi * 3 / 4
    b = .7

    n = np.arange(200)
    c_n = r ** n * np.sin(n * eta) / b
    h_n = c_n
    c = np.roll(c_n, 1)
    c[0] = 0
    c_n += (-2) * c
    c = np.roll(c, 1)
    c[0] = 0
    h_n += 2.25 * c
    c = np.roll(c, 1)
    c[0] = 0
    h_n += (-1.25) * c

    # plt.stem(h_n, use_line_collection=True)
    # plt.ylabel('Impulse Response $h[n]$')
    # plt.xlabel('$n$')

    y_n = signal.convolve(x_n, h_n)

    plt.figure()
    # plt.plot(n, x_n, color='g', label='Input')
    plt.plot(n, c_n, color='k', label='subChannel')
    plt.plot(n, h_n, color='b', label='Channel')
    # plt.plot(n, y_n, color='r', label='Output')
    plt.legend()
    plt.xlabel('Discrete time')
    plt.ylabel('Amplitude')
    plt.show()

    size_input = 1
    node = 4
    size_output = 1
    myESN = basicESN(size_input=size_input,
                     size_output=size_output,
                     node=node,
                     esn_sparsity=0.04,  # 1~5 %
                     spectral_radius=0.1,  # radius larger, the capability of long memory better
                     esn_random_state=42,  # random seed number 42
                     silent=True)
    esn_weights = np.array([-0.7 + 0.7j, -0.7 - 0.7j, -0.7 + 0.6j, -1 / 2])
    input_weights = np.ones((node, size_input))
    myESN.initialize_weights(input_weights, esn_weights, None, None)

    # --------------------- train ---------------------
    delay = 10
    train_sample = np.zeros(shape=(size_input, N - delay))
    train_label = np.zeros(shape=(size_output, N - delay))

    for samp_idx in range(delay, N):
        train_sample[:, samp_idx - delay] = x_n[samp_idx]
        train_label[:, samp_idx - delay] = y_n[samp_idx - delay]
    # train_dataset = mydataset(train_sample, train_label,
    #                           transform=transforms.Compose([ToTensor()]))

    # --------------------- ESN ---------------------
    myESN.train(train_samples=train_sample,
                train_labels=train_label)

    predict_signal_vectors_train, predict_error_train = (
        myESN.predict(test_samples=train_sample, test_labels=train_label, continuous=False))

    a = 1

    '''
    sample_len = 3
    label_len = 3
    node = 20  # if node is too large, ESN will be over-fit
    forcast = 1
    N = 10000  # total length
    # percent = 0.5  # N*percent points are used for training

    myESN = basicESN(size_input=sample_len,
                     size_output=label_len,
                     node=node,
                     esn_sparsity=0.04,     # 1~5 %
                     spectral_radius=0.4,   # radius larger, the capability of long memory better
                     esn_random_state=42)   # random seed number 42
    '''

    '''
    sample_len = 3
    label_len = 3
    node = 48  # if node is too large, ESN will be over-fit
    forcast = 1
    N = 10000  # total length
    # percent = 0.5  # N*percent points are used for training

    myESN = basicESN(size_input=sample_len,
                     size_output=label_len,
                     node=node,
                     esn_sparsity=0.01,     # 1~5 %
                     spectral_radius=0.1,   # radius larger, the capability of long memory better
                     esn_random_state=42)   # random seed number 42
    '''

