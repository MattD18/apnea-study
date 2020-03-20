'''
'''
import tensorflow as tf
from utils import get_dataset_length

class ApneaDataPipeline():
    '''
    '''
    def __init__(self):
        '''
        '''
        self.input_dataset = None
        self.output_dataset = None

class MLPDataPipeline(ApneaDataPipeline):
    '''
    Will eventually become base class for different configurable pipelines
    '''

    def __init__(self, num_seconds=5):
        '''
        '''
        super(MLPDataPipeline, self).__init__()
        self.input_dataset = None
        self.output_dataset = None
        self.num_seconds = num_seconds

    def fit_transform(self, dataset):
        '''
        '''
        self.input_dataset = dataset
        self.transform()
        return self.output_dataset


    def transform(self):
        '''
        '''
        assert(self.input_dataset)

        lambda_func_zero = lambda x,y : (self.normalize_record(x),y)
        lambda_func_one = lambda x,y :self.resize_record(x,y,self.num_seconds)
        lambda_func_two = lambda x,y : tf.data.Dataset.from_tensor_slices((x,y))
        lambda_func_three = lambda x,y : (x, tf.expand_dims(y,0))

        intermediate_dataset = self.input_dataset.map(lambda_func_zero)
        intermediate_dataset = intermediate_dataset.map(lambda_func_one)
        intermediate_dataset = intermediate_dataset.flat_map(lambda_func_two)
        pos_samples = intermediate_dataset.filter(lambda x,y: y==1) 
        neg_samples = intermediate_dataset.filter(lambda x,y: y==0) 
        pos_n = get_dataset_length(pos_samples)
        neg_n = get_dataset_length(neg_samples)
        assert(neg_n > pos_n)
        intermediate_dataset = pos_samples.concatenate(neg_samples.take(pos_n)).shuffle(2*pos_n)
        self.output_dataset = intermediate_dataset.map(lambda_func_three)

    def normalize_record(self, X):
        normalized_X = (X - tf.math.reduce_mean(X)) / tf.math.reduce_std(X)
        return normalized_X

    def resize_X(self, X, num_seconds = 5):
        num_samples = tf.shape(X)[0]
        feature_len = tf.shape(X)[1]
        new_num_samples = num_samples // num_seconds
        new_feature_len = feature_len * num_seconds
        truncation_index = new_num_samples * num_seconds
        new_X = tf.reshape(X[:truncation_index], (new_num_samples, new_feature_len))
        return new_X

    def resize_y(self, y, num_seconds = 5):
        num_samples = tf.shape(y)[0]
        new_num_samples = num_samples // num_seconds
        new_feature_len = num_seconds
        truncation_index = new_num_samples * num_seconds
        new_y = tf.reshape(y[:truncation_index], (new_num_samples, new_feature_len))
        new_y = tf.math.reduce_mean(new_y, axis =1)
        new_y = tf.math.round(new_y)
        return new_y

    def resize_record(self, X,y, num_seconds=5):
        new_X = self.resize_X(X, num_seconds)
        new_y = self.resize_y(y, num_seconds)
        return (new_X, new_y)



class Conv1DDataPipeline(ApneaDataPipeline):
    '''
    Will eventually become base class for different configurable pipelines
    '''

    def __init__(self, num_seconds=5):
        '''
        '''
        super(Conv1DDataPipeline, self).__init__()
        self.input_dataset = None
        self.output_dataset = None
        self.num_seconds = num_seconds

    def fit_transform(self, dataset):
        '''
        '''
        self.input_dataset = dataset
        self.transform()
        return self.output_dataset


    def transform(self):
        '''
        '''
        assert(self.input_dataset)

        lambda_func_zero = lambda x,y : (self.normalize_record(x),y)
        lambda_func_one = lambda x,y :self.resize_record(x,y,self.num_seconds)
        lambda_func_two = lambda x,y : tf.data.Dataset.from_tensor_slices((x,y))
        lambda_func_three = lambda x,y : (tf.expand_dims(x,1), tf.expand_dims(y,0))

        intermediate_dataset = self.input_dataset.map(lambda_func_zero)
        intermediate_dataset = intermediate_dataset.map(lambda_func_one)
        intermediate_dataset = intermediate_dataset.flat_map(lambda_func_two)
        pos_samples = intermediate_dataset.filter(lambda x,y: y==1) 
        neg_samples = intermediate_dataset.filter(lambda x,y: y==0) 
        pos_n = get_dataset_length(pos_samples)
        neg_n = get_dataset_length(neg_samples)
        assert(neg_n > pos_n)
        intermediate_dataset = pos_samples.concatenate(neg_samples.take(pos_n)).shuffle(2*pos_n)
        self.output_dataset = intermediate_dataset.map(lambda_func_three)

    def normalize_record(self, X):
        normalized_X = (X - tf.math.reduce_mean(X)) / tf.math.reduce_std(X)
        return normalized_X

    def resize_X(self, X, num_seconds = 5):
        num_samples = tf.shape(X)[0]
        feature_len = tf.shape(X)[1]
        new_num_samples = num_samples // num_seconds
        new_feature_len = feature_len * num_seconds
        truncation_index = new_num_samples * num_seconds
        new_X = tf.reshape(X[:truncation_index], (new_num_samples, new_feature_len))
        return new_X

    def resize_y(self, y, num_seconds = 5):
        num_samples = tf.shape(y)[0]
        new_num_samples = num_samples // num_seconds
        new_feature_len = num_seconds
        truncation_index = new_num_samples * num_seconds
        new_y = tf.reshape(y[:truncation_index], (new_num_samples, new_feature_len))
        new_y = tf.math.reduce_mean(new_y, axis =1)
        new_y = tf.math.round(new_y)
        return new_y

    def resize_record(self, X,y, num_seconds=5):
        new_X = self.resize_X(X, num_seconds)
        new_y = self.resize_y(y, num_seconds)
        return (new_X, new_y)




