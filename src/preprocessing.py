'''
module to perform ETL on apnea data
'''
import os 
import re

import numpy as np
import tensorflow as tf
import boto3
import pyedflib
import xml.etree.ElementTree as ET

class PreprocessRecords():
    '''
    '''
    def __init__(self, tf_record_dir = 'data/preprocessed_data/', s3_bucket_name=None):
        self.s3_bucket_name = s3_bucket_name
        self.tf_record_dir = tf_record_dir
        self.tf_record_regex = re.compile("(?P<record_name>.*)\.tfrecord")

    def write_to_tf_records_to_local(self, X, y):
        '''
        '''
        for record in X.keys():
            assert X[record].shape[0] == y[record].shape[0]
            serialized_features = tf.io.serialize_tensor(X[record])
            serialized_labels = tf.io.serialize_tensor(y[record])
            serialized_data = {
                'features':tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_features.numpy()])),
                'labels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_labels.numpy()]))
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=serialized_data))
            with tf.io.TFRecordWriter(os.path.join(self.tf_record_dir, f"{record}.tfrecord")) as writer:
                example = example_proto.SerializeToString()
                writer.write(example)

    def read_from_tf_records_from_local(self):
        '''
        '''
        filenames = [filename for filename in os.listdir(self.tf_record_dir) if self.tf_record_regex.match(filename)]
        filenames = [os.path.join(self.tf_record_dir,filename) for filename in filenames]
        raw_dataset = tf.data.TFRecordDataset(filenames)
        parsed_record_dataset = raw_dataset.map(self.read_tfrecord)
        return parsed_record_dataset

    def read_tfrecord(self, serialized_example):
        feature_description = {
            'features': tf.io.FixedLenFeature((),dtype= tf.string),
            'labels': tf.io.FixedLenFeature((),dtype= tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
    
        features =  tf.io.parse_tensor(example['features'], out_type = tf.float64)
        labels =  tf.io.parse_tensor(example['labels'], out_type = tf.float64)

        return features, labels


class EDFLoader():
    '''
    class to process and extract signals from directory of edf files

    Attributes:
    ------------
    edf_dir : str
        location of edf directory


    Methods:
    --------------
    '''
    def __init__(self, edf_dir='data/raw_data/edfs', s3_bucket_name=None):
        self.edf_dir = edf_dir
        self.s3_bucket_name = s3_bucket_name
        self.edf_file_regex = re.compile("(?P<record_name>.*)\.edf")
        #Note which ecg channels should be used?? Question for doctor
        self.ecg_channel_regex = re.compile("(?P<channel_name>ECG\d|EKG\d)")


    def load_from_local(self):
        '''
        '''
        data = {}
        edf_filenames = [filename for filename in os.listdir(self.edf_dir) if self.edf_file_regex.match(filename)]
        for edf_filename in edf_filenames:
            ecg_signal = self.get_ecg_signal_from_file(edf_filename)
            data.update(ecg_signal)
        return data

    def load_from_s3(self):
        '''
        '''
        data = {}
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.s3_bucket_name)
        bucket_objects = bucket.objects.all()
        bucket_keys = [bucket_object.key for bucket_object in bucket_objects]
        edf_keys = [bucket_key for bucket_key in bucket_keys if self.edf_file_regex.match(bucket_key)]
        for edf_key in edf_keys[:2]:
            edf_filename = edf_key.split('/')[-1]
            full_path_filename = os.path.join(self.edf_dir, edf_filename)
            with open(full_path_filename, 'wb') as f:
                bucket.download_fileobj(edf_key, f)
                ecg_signal = self.get_ecg_signal_from_file(edf_filename)
                data.update(ecg_signal)        
                os.remove(full_path_filename)
        return data

    def get_ecg_signal_from_file(self, edf_filename):   
        '''
        '''
        full_edf_file_path = os.path.join(self.edf_dir, edf_filename)
        f = pyedflib.EdfReader(full_edf_file_path)
    
        channel_list = f.getSignalLabels()
        ecg_channel_name = [channel for channel in channel_list if self.ecg_channel_regex.match(channel)][0]
        ecg_channel = channel_list.index(ecg_channel_name)
        ecg_signal = f.readSignal(ecg_channel )
        ecg_freq = f.getSampleFrequency(ecg_channel)
        f._close()
 
        num_seconds = ecg_signal.shape[0] // ecg_freq
        ecg_signal = ecg_signal.reshape(num_seconds, ecg_freq)
        record_name = self.edf_file_regex.match(edf_filename).groupdict()['record_name']
        return {record_name : ecg_signal}




class AnnotationLoader():
    '''
    class to process and extract apnea labels from directory of annotation files

    Attributes:
    ------------
    annotation_dir : str
        location of edf directory

    '''
    def __init__(self, annotation_dir='data/raw_data/annotations-events-nsrr/', s3_bucket_name=None):
        self.annotation_dir = annotation_dir
        self.s3_bucket_name = s3_bucket_name
        self.annotation_file_regex = re.compile("(?P<record_name>.*)-nsrr\.xml")
        self.apnea_event_regex = re.compile('.*apnea.*')

    def load_from_local(self):
        '''
        '''
        data = {}
        annotation_filenames = [filename for filename in os.listdir(self.annotation_dir) \
                                if self.annotation_file_regex.match(filename)]
        for annotation_filename in annotation_filenames:
            annotation = self.get_annotation_from_file(annotation_filename)
            data.update(annotation)
        return data

    def load_from_s3(self):
        '''
        '''
        data = {}
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.s3_bucket_name)
        bucket_objects = bucket.objects.all()
        bucket_keys = [bucket_object.key for bucket_object in bucket_objects]
        annotation_keys = [bucket_key for bucket_key in bucket_keys if self.annotation_file_regex.match(bucket_key)]
        for annotation_key in annotation_keys[:2]:
            annotation_filename = annotation_key.split('/')[-1]
            full_path_filename = os.path.join(self.annotation_dir, annotation_filename)
            with open(full_path_filename, 'wb') as f:
                bucket.download_fileobj(annotation_key, f)
                ecg_signal = self.get_annotation_from_file(annotation_filename)
                data.update(ecg_signal)        
                os.remove(full_path_filename)
        return data



    def get_annotation_from_file(self, annotation_filename):
        '''
        '''
        full_annotation_file_path = os.path.join(self.annotation_dir, annotation_filename)
        root = ET.parse(full_annotation_file_path).getroot()
        scored_events = root.find('ScoredEvents').findall('ScoredEvent')
        apnea_events = [event for event in scored_events \
                        if self.apnea_event_regex.match(event.find('EventConcept').text)]
        apnea_event_indices = [self.get_event_indices(x) for x in apnea_events]
        start_event = [event for event in scored_events \
                       if event.find('EventConcept').text =='Recording Start Time'][0]
        num_seconds = int(float(start_event.find('Duration').text))
        annotation = self.get_annotation_vector(apnea_event_indices, num_seconds)
        record_name = self.annotation_file_regex.match(annotation_filename).groupdict()['record_name']
        return {record_name : annotation}

    def get_event_indices(self, apnea_event):
        '''
        '''
        start = float(apnea_event.find('Start').text)
        duration = float(apnea_event.find('Duration').text)
        end = start + duration
        return (int(start), int(end))

    def get_annotation_vector(self, apnea_event_indices, num_seconds):
        '''
        '''
        annotation_vector = np.zeros(num_seconds)
        for i in apnea_event_indices:
            annotation_vector[i[0]:i[1]] = 1
        return annotation_vector 