import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms

from Dataset import mydataset
from Baseline_RNN.rnn_model import rnnmodel
from Baseline_LSTM.LSTM import lstmmodel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class RNNet:

    def __init__(self, lstm_flag=False, input_dim=28, hidden_dim=128,
                 layer_dim=1, output_dim=10,
                 epochs=30, batch_size=4, lr=3e-4):

        # RNN dimension information
        self.input_dim  = input_dim             # 输入维度
        self.hidden_dim = hidden_dim            # RNN神经元个数
        self.layer_dim  = layer_dim             # RNN的层数
        self.output_dim = output_dim            # 输出维度
        if lstm_flag:
            self.MyRNNimc = rnnmodel(input_dim, hidden_dim, layer_dim, output_dim)
        else:
            self.MyRNNimc = lstmmodel(input_dim, hidden_dim, layer_dim, output_dim)
        # print(self.MyRNNimc)

        # training information
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def train(self, train_dataloader, test_dataloader):
        optimizer = optim.AdamW(self.MyRNNimc.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        train_loss_all = []
        test_loss_all = []
        self.MyRNNimc.to(device)    # GPU

        for epoch in range(self.epochs + 1):
            # print("Epoch {}/{}".format(epoch, self.epochs - 1))
            self.MyRNNimc.train()  # 模式设为训练模式

            train_num, train_loss = 0, 0
            for batch_idx, item in enumerate(train_dataloader):
                # input_size=[batch, time_step, input_dim]
                b_x, b_y = item['sample'].to(device), item['label'].to(device)

                optimizer.zero_grad()                           #
                output = self.MyRNNimc(b_x)
                loss = loss_fn(output, b_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * b_x.size(0)
                train_num += b_x.size(0)
            train_loss_all.append(train_loss / train_num)
            # print("{}, Train Loss: {:.4f}".format(epoch, train_loss_all[-1]))

            if epoch % 10 != 0:
                continue

            # 每10个epoch后,在测试集上测试损失和精度
            test_loss, _ = self.predict(test_dataloader)
            test_loss_all.append(test_loss)
            # print("{} Test Loss: {:.4f}".format(epoch, test_loss_all[-1]))
        torch.save(self.MyRNNimc, "./Baseline_RNN/RNNimc.pkl")
        return train_loss_all, test_loss_all

    def train_figure(self, train_loss_all, test_loss_all):
        plt.figure(figsize=[14, 5])
        plt.plot(train_loss_all, "ro-", label="Train Loss")
        plt.plot(range(0, self.epochs + 1, 10), test_loss_all, "bs-", label="Val Loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("Loss")

    def predict(self, test_dataloader):
        self.MyRNNimc.eval()  # frozen
        test_num, test_loss = 0, 0
        loss_fn = nn.MSELoss()

        output_all = np.zeros(shape=(self.output_dim,
                                     test_dataloader.dataset.samples.shape[1]))
        for batch_idx, item in enumerate(test_dataloader):
            b_x, b_y = item['sample'].to(device), item['label'].to(device)
            output = self.MyRNNimc(b_x)
            # output_all[:, batch_idx] = output.to("cpu").detach().numpy()
            output_all[:, batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size] \
                = output.to("cpu").detach().numpy().T
            loss = loss_fn(output, b_y)

            test_loss += loss.item() * b_x.size(0)
            test_num += b_x.size(0)
        return test_loss / test_num, output_all

