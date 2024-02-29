import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy.linalg import inv  # multiplicative inverse for a matrix
import scipy
from scipy import signal
from scipy.signal import wiener
from numpy import array
from numpy import empty


# activation function
def identity(x):
    return x


class basicESN:

    def __init__(self, size_input, size_output, node,
                esn_sparsity=0.05, spectral_radius=0.95,
                feedback_link=False, skip_link=False,
                activation=identity,
                esn_random_state=None, silent=False):

        # ESN dimension
        self.size_input = size_input
        self.node = node
        self.size_output = size_output

        # hyperparameters
        self.weight_input = None
        self.weight_reservoir = None
        self.weight_output = None
        self.weight_feedback = None
        self.weight_skip = None
        self.esn_sparsity = esn_sparsity
        self.spectral_radius = spectral_radius
        self.activation = activation

        # optional structure
        self.feedback_link = feedback_link  # feedback link
        self.skip_link = skip_link  # skip link

        # other parameters
        self.silent = silent

        # esn_random_state
        if isinstance(esn_random_state, np.random.RandomState):     # a RandomState object,
            self.esn_random_state = esn_random_state
        elif esn_random_state:                                      # a seed
            try:
                self.esn_random_state = np.random.RandomState(esn_random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:                                                       # None
            self.esn_random_state = np.random.mtrand._rand

        self.lastoutput = None
        self.lastinput = None
        self.laststate = None

    # output

    def initialize_weights(self, input_weights, esn_weights, feedback_weight, skip_weight):
        # reservoir weights
        self.weight_reservoir = np.zeros((self.node, self.node), dtype=complex)

        if esn_weights is not None:
            if esn_weights.ndim == 1 and self.node == esn_weights.shape[0]:     # diagonal elements: no interconnection
                [row, col] = np.diag_indices_from(self.weight_reservoir)
                self.weight_reservoir[row, col] = esn_weights
            elif esn_weights.ndim == 2 and self.node == esn_weights.shape[0] and self.node == esn_weights.shape[1]:     # with interconnection
                self.weight_reservoir = esn_weights
            else:
                print("esn_weights initialization wrong in class ESN.py/basicESN/initialize_weights")
                return
        else:
            w = self.esn_random_state.rand(self.node, self.node)    # random: 0~1
            w[self.esn_random_state.rand(*w.shape) > self.esn_sparsity] = 0  # sparsity
            w[w != 0] -= 0.5  # centralization around 0
            radius = np.max(np.abs(np.linalg.eigvals(w)))           # spectral radius
            if radius < 1e-8:
                print("spectral radius wrong in class ESN.py/basicESN/initialize_weights")
                return
            self.weight_reservoir = w * (self.spectral_radius / radius)         # echo state condition

        # input weights
        if input_weights is not None:
            if input_weights.ndim == 2 and self.node == input_weights.shape[0] and self.size_input == input_weights.shape[1]:
                self.weight_input = input_weights
            else:
                print("input_weights initialization wrong in class ESN.py/basicESN/initialize_weights")
                return
        else:
            self.weight_input = self.esn_random_state.rand(
                self.node, self.size_input) * 2 - 1  # random: -1~1

        # feedback weights
        if feedback_weight is not None:
            self.weight_feedback = feedback_weight
        else:
            self.weight_feedback = self.esn_random_state.rand(
                self.node, self.size_output) * 2 - 1                # random: -1~1

        # skip weights
        if skip_weight is not None:
            self.weight_skip = skip_weight
        else:
            self.weight_skip = self.esn_random_state.rand(
                self.size_output, self.size_input) * 2 - 1          # random: -1~1

    def _forward(self, current_input, before_state, before_output):
        """
        one-step update of ESN

        :param current_input:
        :param before_state:
        :param before_output:
        :return:
        """

        # reservoir state
        if self.feedback_link:
            state_preactive = (np.dot(self.weight_reservoir, before_state) +
                               np.dot(self.weight_input, current_input) +
                               np.dot(self.weight_feedback, before_output))
        else:
            state_preactive = (np.dot(self.weight_reservoir, before_state) +
                               np.dot(self.weight_input, current_input))
        current_state = self.activation(state_preactive)

        # current_state = 0.6 * np.tanh(state_preactive)          # tanh
        # current_state = np.maximum(0, state_preactive) - 0.5    # Relu
        # current_state = 2 / (1 + np.exp(-state_preactive))      # sigmoid
        # current_state = 0.5 * state_preactive

        # output
        if self.skip_link:
            current_output = (np.dot(self.weight_output, current_state) +
                                np.dot(self.weight_skip, current_input))
        else:
            current_output = np.dot(self.weight_output, current_state)
        return current_state, current_output

    def train(self, train_samples, train_labels):
        """
        train ESN

        :param train_samples: self.size_input x T
        :param train_labels: self.size_output x T
        :return:
        """

        if not train_labels.ndim == train_samples.ndim:
            print("dimension wrong in ESN.py/function train")
            return

        # if T==1, transform (size_input,) into (size_input,1):
        if train_samples.ndim < 2:
            train_samples = np.reshape(train_samples, (len(train_samples), -1))
            train_labels = np.reshape(train_labels, (len(train_labels), -1))

        # update ESN states to generate a training set
        if not self.silent:
            print("harvesting states and outputs...")
        esn_states = np.zeros((self.node, train_samples.shape[1]), dtype=complex)      # node x T
        esn_output = np.zeros(train_labels.shape, dtype=complex)
        self.weight_output = np.zeros(shape=(self.size_output, self.node))
        _, esn_output[:, 0] = self._forward(train_samples[:, 0],        # the first slot
                                            np.zeros(esn_states[:, 0].shape, dtype=complex),
                                            np.zeros(train_labels[:, 0].shape, dtype=complex))
        for n in range(1, train_samples.shape[1]):
            esn_states[:, n], esn_output[:, n] = (
                self._forward(train_samples[:, n],
                              esn_states[:, n - 1],
                              train_labels[:, n - 1]))

        # optimize the weights of output
        if not self.silent:
            print("fitting...")
        transient = min(int(train_samples.shape[1] / 10), 100)          # discard T/10 or 100 slots
        self.weight_output = np.dot(esn_states[:, transient:],
                                    np.linalg.pinv(train_labels[:, transient:] -
                                                   np.dot(self.weight_skip, train_samples[:, transient:]))).T

        # vital parameters need to know for predicting
        train_error = np.sqrt(np.mean((train_labels[:, transient:] -
                                       np.dot(self.weight_output, esn_states[:, transient:]))**2))
        # train_error = np.sqrt(np.mean((esn_output - train_labels)**2))
        self.laststate  = esn_states[:, -1]
        self.lastinput  = train_samples[:, -1]          # can be discarded?
        self.lastoutput = train_labels[:, -1]
        return train_error

    def predict(self, test_samples, test_labels, continuous=True):

        # if T==1, transform (size_input,) into (size_input,1):
        if test_samples.ndim < 2:
            inputs = np.reshape(test_samples, (len(test_samples), -1))

        time_slots = test_samples.shape[1]

        if continuous:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.node)
            lastinput = np.zeros(self.size_input)
            lastoutput = np.zeros(self.size_output)

        output_series = np.zeros(shape=(self.size_output, time_slots), dtype=complex)
        for slot in range(0, time_slots):
            # print('slot: %d\n' % slot)
            state, output = self._forward(test_samples[:, slot], laststate, lastoutput)
            output_series[:, slot] = output
            laststate, lastoutput = state, output

        # error
        predict_error = np.sqrt(np.mean((output_series - test_labels) ** 2))
        return output_series, predict_error

