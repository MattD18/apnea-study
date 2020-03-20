'''
Module for training loops
'''

import os
import datetime
import tensorflow as tf

class Experiment():
    '''
    '''
    def __init__(self, data, pipeline, model, 
                 optimizer, 
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 epochs=10,
                 batch_size=8,
                 validation_split=.2):
        '''
        '''
        self.raw_data = data
        self.data = None
        self.val_data = None
        self.train_data = None
        self.pipeline = pipeline
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.training_history = None
        self.params= {'num_epochs':epochs, 'batch_size':batch_size, 'validation_split':validation_split}

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
        self.model(tf.stack([record[0] for record in self.data.take(self.params['batch_size'])]))

        if self.params['validation_split']:
            #turn get_legnth_dataset into utils function
            n = self.pipeline.get_dataset_length(self.data)
            num_train = int((1 - self.params['validation_split']) * n)
            self.train_data = self.data.take(num_train)
            self.val_data = self.data.skip(num_train)
        else:
            self.train_data =self.data

        self.training_history = self.model.fit(self.train_data.batch(self.params['batch_size']), 
                                               validation_data=self.val_data.batch(self.params['batch_size']),
                                               epochs=self.params['num_epochs'], 
                                               callbacks=[tensorboard_callback])

    def process_data(self):
        self.data = self.pipeline.fit_transform(self.raw_data)

    def run(self):
        self.process_data()
        self.train_model()

    def save(self):
        pass
