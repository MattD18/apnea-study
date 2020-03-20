'''
Module for training loops
'''

import os
import datetime
import tensorflow as tf

from src.utils import get_dataset_length


class Experiment():
    '''
    '''
    def __init__(self, data, pipeline, model, optimizer, 
                 training_params):
        '''
        '''
        self.raw_data = data
        self.data = None
        self.val_data = None
        self.train_data = None
        self.pipeline = pipeline
        self.model = model
        self.optimizer = optimizer
        #assumes binary classification task
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.training_params= training_params
        self.training_history = None

    def train_model(self):
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%s")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/gradient_tape/dev_test/{ts}', 
                                                            histogram_freq=0,
                                                            write_graph=False, 
                                                            write_images=False,
                                                            update_freq='epoch')
        acc_metric = tf.keras.metrics.BinaryAccuracy( name='Accuracy')
        precision_metric = tf.keras.metrics.Precision( name='Precision')
        recall_metric = tf.keras.metrics.Recall( name='Recall')

        #build model
        self.model.compile(self.optimizer, self.loss, metrics=[acc_metric, precision_metric, recall_metric])
        self.model(tf.stack([record[0] for record in self.data.take(self.training_params['batch_size'])]))

        if self.training_params['validation_split']:
            n = get_dataset_length(self.data)
            num_train = int((1 - self.training_params['validation_split']) * n)
            self.train_data = self.data.take(num_train)
            self.val_data = self.data.skip(num_train)
        else:
            self.train_data =self.data

        self.training_history = self.model.fit(self.train_data.batch(self.training_params['batch_size']), 
                                               validation_data=self.val_data.batch(self.training_params['batch_size']),
                                               epochs=self.training_params['num_epochs'], 
                                               callbacks=[tensorboard_callback])

    def process_data(self):
        self.data = self.pipeline.fit_transform(self.raw_data)

    def run(self):
        self.process_data()
        self.train_model()

    def save(self):
        pass
