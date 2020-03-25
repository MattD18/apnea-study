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
                 val_data=None,
                 log_dir='dev_test',
                 weights_dir = None):
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
        self.log_dir = log_dir
        self.weights_dir = weights_dir

    def train_model(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%s")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/gradient_tape/{self.log_dir}/{self.timestamp}', 
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

        if self.weights_dir:
            self.load_model_weights(self.weights_dir)

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


    def load_model_weights(self, weights_dir):
        '''
        Load model weights from weights_dir
        see demo: https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=OOSGiSkHTERy
        '''
        self.model.load_weights(weights_dir)