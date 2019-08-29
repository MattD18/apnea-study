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
