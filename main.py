'''
main script to train model
'''

import datetime
import time

import tensorflow as tf

import lib.utils as utils
from lib.models import BaseRNN, BaseLSTM, DoubleLSTM
from lib.train import train, evaluate


def main():
    session_start = time.time()

    set_parameters_start = time.time()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_name = 'more_data_test' +  '_' + current_time
    optimizer_params = {'learning_rate':0.0001}
    training_params = {'num_epochs':10,
                       'batch_size':256,
                       'apnea_weight':10}
    preprocess_params = {'featurize_func' : utils.featurize_2,
                         'seq_len':60,
                         'pulse_sample_rate':16,
                         'data_stride':15,
                         'split_seed':2}
    preprocess_func = utils.preprocess_data
    model_params = {'rnn_hidden_dim':20}
    test_split = .15
    val_split = 0.1765
    log_dir = 'logs/gradient_tape/' + session_name
    model_weights_dir = 'model_weights/' + session_name
    print("set session parameters in {}".format(time.time() - set_parameters_start))

    load_data_start = time.time()
    dataset = utils.load_tfrecords(tf_rec_data_dir='data/processed_data')
    num_records = 0
    for i in dataset:
        num_records += 1
    print("data loaded in {}".format(time.time() - load_data_start))

    process_data_start = time.time()
    train_data, test_data = utils.split_dataset(dataset, test_split,seed=preprocess_params['split_seed'])
    train_data, val_data = utils.split_dataset(train_data, val_split,seed=preprocess_params['split_seed'])
    train_data = preprocess_func(train_data,
                                 featurize_func = preprocess_params['featurize_func'],
                                 seq_len=preprocess_params['seq_len'],
                                 pulse_sample_rate=preprocess_params['pulse_sample_rate'],
                                 data_stride=preprocess_params['data_stride'])
    val_data = preprocess_func(val_data,
                               featurize_func = preprocess_params['featurize_func'],
                               seq_len=preprocess_params['seq_len'],
                               pulse_sample_rate=preprocess_params['pulse_sample_rate'],
                               data_stride=preprocess_params['seq_len'])
    test_data = preprocess_func(test_data,
                                featurize_func = preprocess_params['featurize_func'],
                                seq_len=preprocess_params['seq_len'],
                                pulse_sample_rate=preprocess_params['pulse_sample_rate'],
                                data_stride=preprocess_params['seq_len'])
    train_num = train_data._tensors[0].shape[0]
    train_bal = train_data._tensors[1].numpy().mean()
    val_num = val_data._tensors[0].shape[0]
    val_bal = val_data._tensors[1].numpy().mean()
    test_num = test_data._tensors[0].shape[0]
    test_bal = test_data._tensors[1].numpy().mean()
    data_bal = {'train':train_bal,'val':val_bal,'test':test_bal}
    data_size = {'train':train_num,'val':val_num,'test':test_num}
    print("data processed in {}".format(time.time() - process_data_start))


    train_start = time.time()
    if tf.test.is_gpu_available():
        print('Using GPU')
        with tf.device("gpu:0"):
            model = BaseLSTM(rnn_hidden_dim=model_params['rnn_hidden_dim'])
            loss_object = tf.keras.losses.BinaryCrossentropy()
            optimizer = tf.keras.optimizers.Adam(optimizer_params['learning_rate'])
    else:
        model = BaseLSTM(rnn_hidden_dim=model_params['rnn_hidden_dim'])
        loss_object = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(optimizer_params['learning_rate'])
    train(model, train_data,val_data, loss_object, optimizer, log_dir,model_weights_dir,
      num_epochs=training_params['num_epochs'],
      batch_size=training_params['batch_size'],
      apnea_weight=training_params['apnea_weight'])
    train_res = evaluate(model, train_data)
    test_res = evaluate(model, test_data)
    print(train_res)
    print(test_res)
    print("model trained in {}".format(time.time() - train_start ))
    utils.save_session(session_name,
                       model,
                       model_params,
                       num_records,
                       test_split,
                       val_split,
                       preprocess_func,
                       preprocess_params,
                       data_bal,
                       data_size,
                       training_params,
                       optimizer,
                       optimizer_params,
                       train_res,
                       test_res,
                       log_dir,
                       model_weights_dir,
                       res_path = 'results')
    print("Saving session results, session ran in {}".format(time.time() - session_start))



if __name__ == '__main__':
    main()
