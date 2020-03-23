import tensorflow as tf
from tensorflow.keras import Model


class BaseMLP(Model):
    '''
    '''
    def __init__(self):
        super(BaseMLP, self).__init__()
        self.hidden_layer_one = tf.keras.layers.Dense(256, activation='relu')
        self.hidden_layer_two = tf.keras.layers.Dense(128, activation='relu')
        self.hidden_layer_three = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.hidden_layer_one(x)
        x = self.hidden_layer_two(x)
        x = self.hidden_layer_three(x)
        x = self.output_layer(x)
        return x

    
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


class BaseConv1D(Model):
    '''
    '''
    def __init__(self):
        super(BaseConv1D, self).__init__()
        self.conv_one = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16)
        self.avg_pool_one = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.relu_one = tf.keras.layers.ReLU()
        
        self.conv_two = tf.keras.layers.Conv1D(filters=4, kernel_size=32, strides=16)
        self.avg_pool_two = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.relu_two = tf.keras.layers.ReLU()
        
        self.flatten_layer = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        out = self.conv_one(x)
        out = self.avg_pool_one(out)
        out = self.relu_one(out)
        out = self.conv_two(x)
        out = self.avg_pool_two(out)
        out = self.relu_two(out)
        out = self.flatten_layer(out)
        out = self.output_layer(out)
        return out



class RegConv1D(Model):
    '''

    TODO, allow model to take parameters as inputs
    '''
    def __init__(self):
        super(RegConv1D, self).__init__()
        self.conv_one = tf.keras.layers.Conv1D(filters=16, kernel_size=64, strides=16,
                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.avg_pool_one = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.relu_one = tf.keras.layers.ReLU()
        
        self.conv_two = tf.keras.layers.Conv1D(filters=4, kernel_size=32, strides=16,
                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.avg_pool_two = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.relu_two = tf.keras.layers.ReLU()
        
        self.flatten_layer = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))

    def call(self, x):
        out = self.conv_one(x)
        out = self.avg_pool_one(out)
        out = self.relu_one(out)
        out = self.conv_two(x)
        out = self.avg_pool_two(out)
        out = self.relu_two(out)
        out = self.flatten_layer(out)
        out = self.output_layer(out)
        return out