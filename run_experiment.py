'''
Script used to run experiment based on specified configuration
'''

import os
import argparse
import yaml

from tensorflow.keras.optimizers import Adam

from src.etl import RecordETL
from src.preprocess import *
from src.models import *
from src.train import Experiment
from src.utils import load_config_file, get_dataset_length

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('config_path', type=str, help='path to config yaml file')
parser.add_argument('--train_data', type=str, help='path to train data',
                    default='data/processed_data/dev_test_train/')
parser.add_argument('--val_data', type=str, help='path to val data',
                    default='data/processed_data/dev_test_val/')
parser.add_argument('--model_save_dir', type=str, help='path to dir to save model weights',
                    default='model_weights/dev_test')
                    



#pipeline must be referenced here to be used in config file
PIPELINE_DICT = {'Conv1DDataPipeline':Conv1DDataPipeline}

#model must be referenced here to be used in config file
MODEL_DICT = {'BaseConv1D':BaseConv1D}

#optimizer must be referenced here to be used in config file
OPTIMIZER_DICT = {'Adam':Adam}

args = parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config_file(args.config_path)
    train_data_path = args.train_data
    val_data_path = args.val_data
    sav_dir = args.model_save_dir

    #TODO add train and val capabilities
    print('Loading Raw Data')
    etl =  RecordETL(config['data']['train_dir'])
    
    raw_train_dataset = etl.read_from_tf_records_from_local(train_data_path)
    raw_val_dataset = etl.read_from_tf_records_from_local(val_data_path)

    pipeline = PIPELINE_DICT[config['pipeline']['name']](**config['pipeline']['params'])
    model = MODEL_DICT[config['model']['name']](**config['model']['params'])
    optimizer = OPTIMIZER_DICT[config['training']['optimizer']['name']](**config['training']['optimizer']['params'])
    #need to validate on validation set instead of validation split
    experiment = Experiment(train_data=raw_train_dataset,
                            pipeline=pipeline,
                            model=model,
                            optimizer=optimizer,
                            training_params=config['training']['training_loop']['params'],
                            val_data=raw_val_dataset)
    experiment.run()
    experiment.save(sav_dir)

