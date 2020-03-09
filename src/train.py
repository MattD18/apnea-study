'''
Module for training loops
'''

import os

import tensorflow as tf
import numpy as np
from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             accuracy_score,
                             roc_auc_score)

from lib.utils import split_dataset


def train(model, train_data,  val_data, loss_object, optimizer, log_dir,model_weights_dir,
          num_epochs=10, batch_size=16, apnea_weight=1.0):
    """
    training loop for a given model and dataset for apnea second classification,
    tracks loss and accuracy

    Parameters:
    model (tensorflow.keras.Model) : deep classifier for apnea second classification
    train_data (tf.data.Dataset) : datapoints have features of dimension seq_len x
                                sample_rate and 1/0 apnea labels
    val_data (tf.data.Dataset) : datapoints have features of dimension seq_len x
                                    sample_rate and 1/0 apnea labels
    loss_object (tf.keras.losses.Loss) : loss used in training
    optimizer (tf.keras.optimizers.Optimizer) : optimizer used in training
    log_dir (str) : directory to write training results
    model_weights_dir (str) : directory to write model class weights
    num_epochs (int) : number of epochs to train for
    batch_size (int) : batch size for minibatching
    apnea_weight (float) : weight to assign positive labels during training

    """
    train_size = train_data._tensors[0].shape[0]
    val_size = val_data._tensors[0].shape[0]

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.BinaryAccuracy(name = 'train_acc',threshold = .5)
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc =tf.keras.metrics.BinaryAccuracy(name = 'val_acc',threshold = .5)

    train_log_dir = os.path.join(log_dir,'train')
    val_log_dir = os.path.join(log_dir,'val')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    print("Starting Training")
    for epoch in range(num_epochs):
        train_data = train_data.shuffle(train_size, seed=42)

        #train model using minibatches
        for x_batch,y_batch in train_data.batch(batch_size, drop_remainder=True):
            with tf.GradientTape() as tape:
                predictions = tf.expand_dims(tf.reshape(model(x_batch),[-1]),1)
                labels = tf.expand_dims(tf.reshape(y_batch,[-1]),1)
                sample_weight = tf.convert_to_tensor((labels.numpy()*apnea_weight)+1)
                loss = loss_object(labels, predictions, sample_weight=sample_weight)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss(loss)
            train_acc.update_state(labels,predictions)
        #write epoch train results to log
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('acc', train_acc.result(), step=epoch)

        #evaluate model perforamnce on validation set at each epoch
        for x_val, y_val in val_data.batch(batch_size, drop_remainder=False):
            predictions = tf.expand_dims(tf.reshape(model(x_batch),[-1]),1)
            labels = tf.expand_dims(tf.reshape(y_batch,[-1]),1)
            sample_weight = tf.convert_to_tensor((labels.numpy()+1)*apnea_weight)
            loss = loss_object(labels, predictions, sample_weight=sample_weight)
            val_loss(loss)
            val_acc.update_state(labels,predictions)
        #write epoch val results to log
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('acc', val_acc.result(), step=epoch)

        # Reset metrics every epoch
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()

    model.save_weights(os.path.join(model_weights_dir,'weights.ckpt'))

def evaluate(model, dataset):
    """
    Evaluate model on sleep apnea dataset

    Parameters:
    model (tensorflow.keras.Model) : deep classifier for apnea second classification
    dataset (tf.data.Dataset) : datapoints have features of dimension seq_len x
                                sample_rate and 1/0 apnea labels

    Return:
    eval_res (dict) : dictionary of accuracy, recall, precision, f1, and auc
    """
    n = dataset._tensors[0].shape[0]
    for x, y in dataset.batch(n,drop_remainder=False):
        y_pred_probs = tf.reshape(model(x),[-1]).numpy()
        y_true = tf.reshape(y,[-1]).numpy()
    y_pred = np.round(y_pred_probs)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred , average="binary")
    precision = precision_score(y_true, y_pred , average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true,y_pred_probs)

    eval_res = {'accuracy' : accuracy,
                'recall' : recall,
                'precision' : precision,
                'f1' : f1,
                'auc': auc}

    return eval_res
