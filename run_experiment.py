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


    #TODO add train and val capabilities
    print('Loading Raw Data')
    etl =  RecordETL(config['data']['train_dir'])
    raw_dataset = etl.read_from_tf_records_from_local().take(8)
    test_raw_dataset = etl.read_from_tf_records_from_local().take(10).skip(8)

    pipeline = PIPELINE_DICT[config['pipeline']['name']](**config['pipeline']['params'])
    model = MODEL_DICT[config['model']['name']](**config['model']['params'])
    optimizer = OPTIMIZER_DICT[config['training']['optimizer']['name']](**config['training']['optimizer']['params'])
    #need to validate on validation set instead of validation split
    experiment = Experiment(data=raw_dataset,
                            pipeline=pipeline,
                            model=model,
                            optimizer=optimizer,
                            training_params=config['training']['training_loop']['params'])
    experiment.run()


