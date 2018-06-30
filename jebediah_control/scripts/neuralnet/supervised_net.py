#! /usr/bin/env python
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import optimizers

import time
import os
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class neuralNet(object):
    def __init__(self, *args):
        pd.options.display.max_rows = 8
        pd.options.display.float_format = '{:.4f}'.format

    def Load_Data(self, file):
        """Get gait data from file
        
        Arguments:
            file {string} -- path to file
        """

        self.data = pd.read_csv(file, sep=' ')
        # print self.data
    
    def linear_scale(self, series, min_scale=-1, max_scale=1, min_val=None, max_val=None):
        """Scale value to a range linearly
        The scale range can be set manualy and the min and max values do be mapped by default
        will be get from max/min values of a series, but it also can be set manually. 

        Args:
            series: Pandas series to be scaled.
            min_scale: Start value of scale. Default = -1.
            max_scale: End value of scale. Default = 1
            min_val: Min value to be scaled. Default = series.min()
            max_val: Max value to be scaled. Default = series.max()
            
        Returns:
            Scaled pandas series.
        """
        if min_val is None: min_val = series.min()
        if max_val is None: max_val = series.max()
        print "%15s"%series.name, " mapped: From", min_val, "-", max_val, " to ", min_scale, "-", max_scale
        scale = (max_val - min_val) / (max_scale - min_scale)
        return series.apply(lambda x:((x - min_val) / scale) - (max_scale - min_scale)/2)

    def Preprocess(self):
        """Preprocess and syntesize data for neural net
        """
        # SINTETIC DATA

        # NORMALIZATION (if needed)
        for i in range(12):
            name = "motor_state_%i"%i
            self.data[name] = self.linear_scale(self.data[name], min_val=-90, max_val=90)
            name = "action_%i"%i
            self.data[name] = self.linear_scale(self.data[name], min_val=-90, max_val=90)
        # the other values are already small numbers

            
        # print self.data
        # print self.data.describe()

    def Split(self, validation_proportion=0.15, target_numbers=12):
        """Split the class data into targets and labels for a validation and training and return a dictonary containing all 4 of the sets
        also keep a copy from it into the class.
        
        Keyword Arguments:
            validation_proportion {float} -- [Which proportion of the data will me separated for validation] (default: {0.3})
            target_numbers {int} -- [Number of targets (must all be the end of the dataset)] (default: {12})
        
        Returns:
            [Dictionary] -- [Containing a label 'training' and 'validation' each also containing a label 'labels' and 'targets' which are pandas datasets]
        """

        # Separate into training and validation
        validation_length = int(len(self.data)*validation_proportion)
        training_length = len(self.data) - validation_length
        print "Data distribution: ", 1-validation_proportion, "to ", validation_proportion
        print "Training:    ", training_length
        print "Validation:  ", validation_length
        print "Total:       ", len(self.data)
        training = self.data.head(training_length) 
        validation = self.data.tail(validation_length)
        # Separate target and validation
        column_numbers = len(validation.columns) # keep it variable due to possible sintetic labels
        tra_labels = training.iloc[:,0:column_numbers-target_numbers]
        tra_targets = training.iloc[:,column_numbers-target_numbers:]
        val_labels = validation.iloc[:,0:column_numbers-target_numbers]
        val_targets = validation.iloc[:,column_numbers-target_numbers:]
        self.dataset = {'training':{'labels':tra_labels, 'targets':tra_targets}, 'validation':{'labels':val_labels, 'targets':val_targets}}.copy()
        return self.dataset
    
    def model_relu(self, input = 32, output = 12, learning_rate=0.1):
        self.model_name = "simple_relu"
        model = Sequential()
        model.add(Dense(24, input_dim=input, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(18, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(output, activation='relu'))
        opti = optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=opti, metrics=['accuracy'])
        self.model = model
    
    def fit_model(self, epochs=100, batch=50):
        # create log path
        log_path = os.getcwd()
        log_path = log_path + "/logs/" + self.model_name + time.strftime('_%Y-%m-%d_%H-%M-%S')
        mkdir_p(log_path)
        print "Logging to ", log_path
        # log for tensorboard visualization
        tensorboard = TensorBoard(log_dir=log_path)
        plot_model(self.model, to_file=log_path+'/model.png', show_shapes=True, show_layer_names=True)
        # fit model
        self.model.fit(self.dataset['training']['labels'], self.dataset['training']['targets'], epochs=epochs, batch_size=batch, callbacks=[tensorboard])
        # score model with validating data
        scores = self.model.evaluate(self.dataset['validation']['labels'], self.dataset['validation']['targets'])
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
def main():
    handler = neuralNet()
    handler.Load_Data("Dataset.txt")
    handler.Preprocess()
    handler.Split()
    handler.model_relu()
    handler.fit_model()
    
if __name__ == '__main__':
       main()
        