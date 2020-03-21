'''
Module for training loops
'''

import os
import datetime
import tensorflow as tf

class Experiment():
    '''
    '''
    def __init__(self, train_data, pipeline, model, optimizer, 
                 training_params, 
                 val_data=None):
        '''
        '''
        self.raw_train_data = train_data
        self.raw_val_data = val_data
        self.train_data = None
        self.val_data = None
        self.pipeline = pipeline
        self.model = model
        self.optimizer = optimizer
        #assumes binary classification task
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.training_params= training_params
        self.training_history = None
        self.timestamp = None

    def train_model(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%s")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/gradient_tape/dev_test/{self.timestamp}', 
                                                            histogram_freq=0,
                                                            write_graph=False, 
                                                            write_images=False,
                                                            update_freq='epoch')
        acc_metric = tf.keras.metrics.BinaryAccuracy( name='Accuracy')
        precision_metric = tf.keras.metrics.Precision( name='Precision')
        recall_metric = tf.keras.metrics.Recall( name='Recall')

        #build model
        self.model.compile(self.optimizer, self.loss, metrics=[acc_metric, precision_metric, recall_metric])
        self.model(tf.stack([record[0] for record in self.train_data.take(self.training_params['batch_size'])]))


        self.training_history = self.model.fit(x = self.train_data.batch(self.training_params['batch_size']), 
                                               validation_data=self.val_data.batch(self.training_params['batch_size']),
                                               epochs=self.training_params['num_epochs'], 
                                               callbacks=[tensorboard_callback])

    def process_data(self):
        self.train_data = self.pipeline.fit_transform(self.raw_train_data)
        if self.raw_val_data:
            self.val_data = self.pipeline.fit_transform(self.raw_val_data)



    def run(self):
        self.process_data()
        self.train_model()

    def save(self, save_dir):
        save_path = os.path.join(save_dir, self.timestamp)
        self.model.save_weights(save_path, save_format='tf')
