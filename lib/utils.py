'''
Utility functions for module
'''
import os
import re


import pyedflib
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np


def create_tfrecords_from_raw_data(raw_data_dir='data/raw_data',
                                   tf_rec_data_dir='data/processed_data',
                                   pulse_sample_rate=16,
                                   pulse_signal_number=26,
                                   annotation_event_index=2
                                   ):
    """extracts pulse and apnea annotation data from NSRR edf and annotaion files
    and saves data in tfrecord format

    Parameters:
    raw_data_dir (str) : file path to directory with nsrr files
    tf_rec_data_dir (str) : file path to directory where tfrecords will be written to
    pulse_sample_rate (int) : sample rate of pulse data in Hz
    pulse_signal_number (int) : index of pulse data in edf file
    annotation_event_index (int) : index of scored events in annotation file

    """
    edf_dir = os.path.join(raw_data_dir,'edfs')
    annotation_dir = os.path.join(raw_data_dir,'annotations-events-nsrr/')
    record_names = get_record_names(edf_dir)
    for record in record_names:
        edf_file_path = os.path.join(edf_dir,record+'.edf')
        #extract pulse data from edf file
        edf_data = pyedflib.EdfReader(edf_file_path)
        num_pulse_data_points = edf_data.readSignal(pulse_signal_number).shape[0]
        num_seconds = int(num_pulse_data_points / pulse_sample_rate)
        pulse = edf_data.readSignal(pulse_signal_number)
        #extract annotation data from edf file
        annotation = np.zeros(num_seconds)
        annotation_file_path = os.path.join(annotation_dir,record+'-nsrr.xml')
        annotation_data = ET.parse(annotation_file_path)
        annotation_data_root = annotation_data.getroot()
        scored_events = annotation_data_root[annotation_event_index]
        #store apnea annotations in array that is num_seconds long,
        #each element of array is 1 if apnea was detected during that second
        #of sleep, 0 otherwise
        for event in scored_events:
            if event[1].text is not None:
                if 'Obstructive apnea|Obstructive Apnea' == event[1].text:
                    start = int(float(event[2].text))
                    duration = int(float(event[3].text))
                    end = start + duration + 1
                    annotation[start:end] = 1
        #store pulse and annotation data in a tfrecord
        features_dict = {
            'x':tf.train.Feature(float_list=tf.train.FloatList(value=pulse)),
            'y':tf.train.Feature(float_list=tf.train.FloatList(value=annotation))
        }
        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        #write tfrecord to disk
        tfrecord_file_path = os.path.join(tf_rec_data_dir ,record + '.tfrecord')
        with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
            writer.write(example.SerializeToString())
        #close edf file
        edf_data._close()
        del edf_data

def load_tfrecords(tf_rec_data_dir='data/processed_data'):
    """loads pulse and apnea data in tfrecord format into a tensorflow Dataset

    Parameters:
    tf_rec_data_dir (str) : file path to directory where tfrecords will be written to

    Returns:
    dataset tf.data.Dataset: tensorflow dataset of n records where n is number of patients

    """
    record_names = get_record_names(tf_rec_data_dir)
    tfrecord_path_list = [os.path.join(tf_rec_data_dir,record + '.tfrecord') for record in record_names]
    #load tfrecords into tf dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path_list)
    return dataset

def preprocess_data(dataset,
                    seq_len=30,
                    pulse_sample_rate=16,
                    data_stride=30):
    """
    Preprocessing helper function for load_tfrecords

    Parameters:
    dataset (tf.data.Dataset) : dataset of patient records
    seq_len (int) : length of sequence in output datapoints
    pulse_sample_rate (int) : given sample rate in edf, becomes dimension of feature vector
    data_stride (int) : determines how many datapoints are generated from single patient record

    Returns :
    dataset (tf.data.Dataset) :datapoints have features of dimension seq_len x
                                sample_rate and 1/0 apnea labels

    """
    #preprocessing steps on dataset
    dataset = dataset.map(_to_dense_tensors)
    X = []
    y = []
    for dp in dataset:
        xy = _featurize(dp,
                        seq_len=seq_len,
                        pulse_sample_rate=pulse_sample_rate,
                        data_stride=data_stride)
        X.extend(xy[0])
        y.extend(xy[1])
    dataset = tf.data.Dataset.from_tensor_slices((tf.stack(X),tf.stack(y)))
    return dataset

def _to_dense_tensors(element):
    """
    Preprocessing helper function for load_tfrecords,
    returns features and labels as dense vectors

    Parameters:
    element : element of tf.data.TFRecordDataset

    Returns:
    output_dict : element of tf.data.TFRecordDatasets

    """
    # Create a dictionary describing the features of the tfrecords
    feature_description = {
        'x': tf.io.VarLenFeature(tf.float32),
        'y': tf.io.VarLenFeature(tf.float32)
    }
    data = tf.io.parse_single_example(element,feature_description)
    x =  tf.sparse.to_dense(data['x'])
    y =  tf.sparse.to_dense(data['y'])
    output_dict = {'y':y, 'x':x}
    return output_dict

def _featurize(element, seq_len=30, pulse_sample_rate=16, data_stride=30):
    """
    Normalizes pulse data by patient across night of sleep by subtracting mean
    and dividing by standard deviation

    Converts single patient record into many datapoints, each datapoint is a
    seq_len length sequence with pule_sample_rate dimensional features and 1
    dimensional labels

    Parameters:
    element : element of tf.data.TFRecordDataset
    seq_len (int) : length of sequence in output datapoints
    pulse_sample_rate (int) : given sample rate in edf, becomes dimension of feature vector
    data_stride (int) : determines how many datapoints are generated from single patient record

    Returns:
    output_tuple : collection of new datapoints


    """
    y = element['y']
    x = element['x']
    X = []
    Y = []
    #normalize pulse data by night of sleep
    x = (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)
    #convert night of sleep from 1 dimensional sequence of num samples length
    #to a sequence of pulse_sample_rate dimension of num seconds length
    num_seconds = x.shape[0]//pulse_sample_rate
    x_trunc = x[:pulse_sample_rate*(num_seconds)]
    x = tf.reshape(x_trunc,[num_seconds,pulse_sample_rate])
    #create new datapoints, according to data_stride
    num_data_points = (x.shape[0]//data_stride) - (seq_len//data_stride)
    for i in range(0,num_data_points):
        X.append(x[data_stride*i:data_stride*i+seq_len])
        Y.append(y[data_stride*i:data_stride*i+seq_len])
    return (X,Y)

def get_record_names(data_dir):
    """Helper function to get record names without file extensions
    from a directory of nsrr records

    Parameters:
    data_dir (str): file path to data directory

    Returns:
    record_names (list of strings): record names in directory

    """
    extension_regex = r"\..*$"
    dir_contents = os.listdir(data_dir)
    #remove extensions
    dir_contents = [re.sub(extension_regex,'',file) for file in dir_contents]
    #keep just records from directory contents
    record_names = list(filter(lambda x : x != '',dir_contents))
    return record_names

def split_dataset(dataset, split_ratio, seed=42):
    """Splits a tensorflow dataset according to split ratio

    Parameters:
    dataset (tf.data.Dataset) : tensorflow dataset
    split_ratio (float) : number between 0 and 1, how to split dataset
    seed (int) : random seed for shuffling dataset


    Returns:
    datasets (tuple of tf.data.Dataset): split datasets
    """
    #necessary because AttributeError: 'TFRecordDatasetV2' object has no attribute '_tensors'
    n = 0
    for i in dataset:
        n+=1
    first_size = int((1 - split_ratio) * n)
    second_size = int(split_ratio * n)

    dataset = dataset.shuffle(n, seed=seed)
    first_data = dataset.take(first_size)
    second_data = dataset.skip(first_size).take(second_size)


    return (first_data, second_data)
