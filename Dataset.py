import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


# auto-regression (AR) process
class stochastic_signal:

    def __init__(self, length,
                 coef=(0.9, -0.4), bias=0, varu=0.0001, varv=0.0001,
                 sample_len=100, label_len=100, forcast=1):
        """

        :param length:
        :param percent:
        :param coef:
        :param bias:
        :param varu:
        :param varv:
        :param sample_len:
        :param label_len:
        :param forcast:
        """
        # AR signal
        self.length = length
        self.percent = None
        self.coef = coef
        self.bias = bias
        self.varu = varu
        self.varv = varv
        self.signal_original = None
        self.signal_distorted = None

        # each sample
        self.sample_len = sample_len
        self.label_len = label_len
        self.forcast = forcast

        # num
        self.total_samples = length - (sample_len + forcast) + 1

    # another way to generate AR
    def generate_ar_process(self, coef, sigma_1, bias, n_samples):
        """
        生成AR过程的信号。

        :param coef: AR系数列表。
        :param sigma_1: 激励噪声的标准差。
        :param bias: AR方程的偏差。
        :param n_samples: 生成的样本数。
        :return: 生成的AR过程样本。
        """
        p = len(coef)
        # 初始化序列
        ar_process = np.zeros(n_samples)
        # 添加激励噪声
        e = np.random.normal(0, sigma_1, n_samples)
        for n in range(p, n_samples):
            ar_process[n] = bias + np.dot(coef, ar_process[n - p:n][::-1]) + e[n]
        return ar_process

    def signal_generator(self):
        self.signal_original = np.zeros(2 * self.length)
        u = np.sqrt(self.varu) * np.random.normal(0, 1, (2 * self.length, 1))

        for n in range(len(self.coef), 2 * self.length):
            self.signal_original[n] = (np.sum(self.signal_original[n-len(self.coef): n] * np.flip(self.coef))
                                       + self.bias) + u[n]
            tmp = 0
            for i in range(len(self.coef)):
                if self.signal_original[n-i-1] >= 0:
                    tmp += np.power(self.signal_original[n - i - 1], (i+1)/4) * self.coef[i]
                else:
                    tmp += self.signal_original[n - i - 1] ** 2 / 10 * self.coef[i]
            self.signal_original[n] = tmp + self.bias + u[n]
        self.signal_original = self.signal_original[self.length:]

        '''
        for n in range(1, self.length - 1):
            self.signal_original[n + 1] = (self.bias + self.coef[0] * self.signal_original[n] +
                                           self.coef[1] * self.signal_original[n - 1] + u[n + 1])      # original
        '''

        signal_noise = np.sqrt(self.varv) * np.random.normal(0, 1, self.length)             # AWGN
        self.signal_distorted = self.signal_original + signal_noise                                 # observed

    def dataset_generator(self, percent):
        self.percent = percent

        if self.label_len < self.forcast:
            print("shape wrong in function main.py/dataset_generator")
            return

        sample = np.zeros(shape=(self.sample_len, self.total_samples))
        label = np.zeros(shape=(self.label_len, self.total_samples))
        shift = self.sample_len + self.forcast - self.label_len

        for samp_idx in range(self.total_samples):
            sample[:, samp_idx] = self.signal_distorted[samp_idx: samp_idx + self.sample_len]
            label[:, samp_idx] = self.signal_distorted[shift + samp_idx: shift + samp_idx + self.label_len]

        train_sample = sample[:, :int(self.total_samples * self.percent)]
        train_label  = label[:, :int(self.total_samples * self.percent)]
        test_sample  = sample[:, int(self.total_samples * self.percent):]
        test_label   = label[:, int(self.total_samples * self.percent):]
        return train_sample, train_label, test_sample, test_label

    def signal_predict(self, predict_signal_vectors_train, predict_signal_vectors_test):
        start = ((self.sample_len + int(self.total_samples * self.percent) - 1) +
                 self.forcast)  # beginning location of prediction

        # trainset
        if predict_signal_vectors_train.ndim < 2:
            predict_signal_vectors_train = np.reshape(predict_signal_vectors_train,
                                                      (len(predict_signal_vectors_train), -1))
        predict_signal_train = np.zeros(start - (self.sample_len + self.forcast - 1))
        for slot in range(predict_signal_vectors_train.shape[1]):
            predict_signal_train[slot] = predict_signal_vectors_train[-1, slot]
        predict_time_train = list(range(self.sample_len + self.forcast - 1, start))

        # testset
        if predict_signal_vectors_test.ndim < 2:
            predict_signal_vectors_test = np.reshape(predict_signal_vectors_test,
                                                     (len(predict_signal_vectors_test), -1))
        predict_signal_test = np.zeros(self.length - start)
        for slot in range(predict_signal_vectors_test.shape[1]):
            predict_signal_test[slot] = predict_signal_vectors_test[-1, slot]
        predict_time_test = list(range(start, self.length))

        return predict_signal_train, predict_time_train, predict_signal_test, predict_time_test


# defined for RNN dataloader
class mydataset(Dataset):

    def __init__(self, samples, labels,
                 transform=None):
        super(mydataset, self).__init__()
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.samples.shape[1]

    def __getitem__(self, idx):
        item = {'sample': self.samples[:, idx], 'label': self.labels[:, idx]}

        if self.transform:
            item = self.transform(item)

        return item

    # if we use real-world sampling data, transform()
    # should be included for normalization


class ToTensor(object):                         # 样本转tensor
    """Convert ndarray in sample to Tensor."""
    def __call__(self, item):
        sample, label = item['sample'], item['label']
        sample = torch.from_numpy(sample.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        return {'sample': sample.unsqueeze(0),
                'label': label.squeeze(0)
                }
