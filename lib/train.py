'''
Module for training loops
'''

import os

import tensorflow as tf

from lib.utils import split_dataset


def train(model, dataset, loss_object, optimizer, log_dir,
          val_split=.2, num_epochs=10, batch_size=16):
    """
    training loop for a given model and dataset for apnea second classification,
    tracks loss and accuracy

    Parameters:
    model (tensorflow.keras.Model) : deep classifier for apnea second classification
    dataset (tf.data.Dataset) : datapoints have features of dimension seq_len x
                                sample_rate and 1/0 apnea labels
    loss_object (tf.keras.losses.Loss) : loss used in training
    optimizer (tf.keras.optimizers.Optimizer) : optimizer used in training
    log_dir (str) : directory to write training results
    val_split (float) : percent of train set to use for validation
    num_epochs (int) : number of epochs to train for
    batch_size (int) : batch size for minibatching

    """
    train_size = int((1-val_split)*dataset._tensors[0].shape[0])
    val_size = int((val_split)*dataset._tensors[0].shape[0])
    train_data, val_data = split_dataset(dataset, val_split)


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
                predictions = tf.reshape(model(x_batch),[-1])
                labels = tf.reshape(y_batch,[-1])
                loss = loss_object(labels, predictions)
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
            predictions = tf.reshape(model(x_val),[-1])
            labels = tf.reshape(y_val,[-1])
            loss = loss_object(labels, predictions)
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
