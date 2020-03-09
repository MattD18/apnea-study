import tensorflow as tf
from tensorflow.keras import Model


class BaseRNN(Model):

    def __init__(self, rnn_hidden_dim=10):
        super(BaseRNN, self).__init__()
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(rnn_hidden_dim),
                                       time_major=False,
                                       return_sequences = True) #(batch, timesteps, ...)
        self.d1 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.rnn(x)
        x = self.d1(x)
        return x


class BaseLSTM(Model):

    def __init__(self, rnn_hidden_dim=10):
        super(BaseLSTM, self).__init__()
        self.lstm = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(rnn_hidden_dim),
                                        time_major=False,
                                        return_sequences = True) #(batch, timesteps, ...)
        self.d1 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.lstm(x)
        x = self.d1(x)
        return x

class DoubleLSTM(Model):

    def __init__(self, rnn_hidden_dim=10):
        super(DoubleLSTM, self).__init__()
        cells = [tf.keras.layers.LSTMCell(rnn_hidden_dim),
                 tf.keras.layers.LSTMCell(rnn_hidden_dim)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(cells)

        self.lstm = tf.keras.layers.RNN(stacked_lstm,
                                        time_major=False,
                                        return_sequences = True) #(batch, timesteps, ...)
        self.d1 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.lstm(x)
        x = self.d1(x)
        return x
