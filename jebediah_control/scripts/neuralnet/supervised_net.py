#! /usr/bin/env python
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import optimizers
from math import sqrt, isnan
import matplotlib.pyplot as plt
import time
import os
import errno
import random
import pandas as pd
import numpy as np

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
        pd.options.display.max_rows = 12
        pd.options.display.float_format = '{:.2f}'.format
        self.log_path = os.getcwd()
        self.log_path = self.log_path + "/logs"
        mkdir_p(self.log_path)
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
    
    def model_dense(self, batch = 50, learning_rate=0.01, optimizer = 2, topology = [[0, 24, 1],[0, 24, 0],[0.1, 24, 1],[0, 24, 1],[0, 12, 1]]):
        """Define a dense keras module
        
        Keyword Arguments:
            batch {training batch size} -- Batch size to use on fit fase (default: {50})
            learning_rate {float} -- NN learning rate (default: {0.01})
            optimizer {int} -- Optimizer:  (default: {2})
            topology {list} -- List of list where each item is a layer and has to contain 3 values [dropout nodes activation] (default: {[[0, 24, 1],[0, 24, 0],[0.1, 24, 1],[0, 24, 1],[0, 12, 1]]})
        """

        self.batch = batch
        # Optimizer
        if optimizer == 1:
            opti = optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0)
        elif optimizer == 2:
            opti = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer == 3:
            opti = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
        elif optimizer == 4:
            opti = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
        elif optimizer == 5:
            opti = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
        elif optimizer == 6:
            opti = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        elif optimizer == 7:
            opti = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # topology
        # [dropout layer_nodes activation] 
        # convert num to activation 
        for i in reversed(range(len(topology))):
            #if activation function is 0 remove layer
            if topology[i][2] == 0: del topology[i]
            elif topology[i][2] == 1: topology[i][2] = 'relu'
            elif topology[i][2] == 2: topology[i][2] = 'softmax'
            elif topology[i][2] == 3: topology[i][2] = 'elu'
            elif topology[i][2] == 4: topology[i][2] = 'selu'
            elif topology[i][2] == 5: topology[i][2] = 'softplus'
            elif topology[i][2] == 6: topology[i][2] = 'softsign'
            elif topology[i][2] == 7: topology[i][2] = 'tanh'
            elif topology[i][2] == 8: topology[i][2] = 'sigmoid'
            elif topology[i][2] == 9: topology[i][2] = 'hard_sigmoid'
            elif topology[i][2] == 10: topology[i][2] = 'linear'
        
        self.model_name = "dense"
        model = Sequential()
        input = len(self.dataset['training']['labels'].columns)
        # [dropout layer_nodes activation type]
        for i in range(len(topology)):
                # force last layer to size of outputs
                if i == len(topology)-1: topology[i][1] = len(self.dataset['training']['targets'].columns)
                model.add(Dense(topology[i][1], input_dim=input, activation=topology[i][2]))
                model.add(Dropout(topology[i][0]))
                input = topology[i][1]
        model.compile(loss='mean_squared_error', optimizer=opti, metrics=['mae', 'accuracy'])
        self.model = model
    
    def fit_model(self, epochs=5, plot=None, verbose=0, log=True, title="Model Fitting"):
        
        if log == True: # create log path
            log_path =  self.log_path + self.model_name + time.strftime('_%Y-%m-%d_%H-%M-%S')
            print "Logging to ", log_path
            mkdir_p(log_path)
            tensorboard = TensorBoard(log_dir=log_path)
            # log for tensorboard visualization
            plot_model(self.model, to_file=log_path+'/model.png', show_shapes=True, show_layer_names=True)
            cb = [tensorboard]
        else:
            cb = []
        # fit model
        history = self.model.fit( self.dataset['training']['labels'], 
                        self.dataset['training']['targets'], 
                        epochs=epochs, 
                        batch_size=self.batch, 
                        callbacks=cb,
                        validation_data=(   self.dataset['validation']['labels'], 
                                            self.dataset['validation']['targets']),
                        verbose = verbose)
        if plot is not None:
            plt.clf()
            plt.suptitle(title)
            # print(history.history.keys())
            # summarize history for accuracy
            plt.subplot(1,2,1)
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.grid()

            # summarize history for loss
            plt.subplot(1,2,2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['Training', 'Validation'], loc='lower right')
            plt.grid()
            if plot == 1:
                plt.draw()
                plt.pause(0.001)
            else:
                plt.show()
        # return metrics
        if isnan(history.history['acc'][-1]) or isnan(history.history['val_loss'][-1]) or isnan(history.history['loss'][-1]):
            return 0, 0
        return history.history['val_acc'][-1]
    
    # EVOLUTION
    def validate_individual(self, vec, bounds, float_index_list):
        """Round integers and bound values within a
        
        Arguments:
            vec {list} -- vector to be validated
            bounds {[type]} -- list of tuples with boundries
            float_index_list {list} -- list of indexes where the vector should not be converted to int
        
        Returns:
            list -- Validated vector
        """

        ret = []
        for i in range(len(vec)):
            # if smaller thand boundary
            if vec[i] < bounds[i][0]: ret.append(bounds[i][0])
            # if larger then boundary
            elif vec[i] > bounds[i][1]: ret.append(bounds[i][1])
            else: ret.append(vec[i])
            # round if needed
            if not(i in float_index_list):
                ret[i] = int(round(ret[i]))
        return ret
    
    def convert_to_model(self, vector):
        batch = vector[0]
        lr = vector[1]
        opti = vector[2]
        topology = []
        layer_num = (len(vector)-3)/3
        for i in range(layer_num):
            topology.append([vector[3+i*3], vector[3+i*3+1], vector[3+i*3+2]])
            i = i + 3
        return batch, lr, opti, topology
    
    def fitness(self, individual, training_epochs, verbosity=0, title="Individual Fitness"):
        b, l, opti, top = self.convert_to_model(individual)
        # create model
        self.model_dense(b, l, opti, top)
        # train and evaluate
        return self.fit_model(epochs=training_epochs, log=False, verbose=verbosity, plot=1, title=title)

    def differential_evolution(self, population_size, mutation_factor, crossover_factor, bounds, float_index_list, epochs, training_epochs=100, file=None):
        ### INITIALIZE POPULATION
        population = []
        fitness_vec = []
        epoch_acc = pd.DataFrame(columns=['Max', 'mean', 'Min'])
        time_before = time.time()
        print "Initializing population."
        if file is None:
            for i in range(0, population_size):
                indv = []
                for j in range(len(bounds)):
                    indv.append(random.uniform(bounds[j][0],bounds[j][1]))
                indv = self.validate_individual(indv, bounds, float_index_list)
                population.append(indv)
                # calculate fitness
                fitness_vec.append(self.fitness(indv, training_epochs))
        else: # read from file but recalculate fitness (method might have changed)
            population = pd.read_csv(file)
            population.drop(population.columns[[0]], axis=1, inplace=True)
            population = population.values.tolist()
            print population
            for indv in population:
                indv = self.validate_individual(indv, bounds, float_index_list)
                fitness_vec.append(self.fitness(indv, training_epochs))
            
        ### Evolutionary loop
        # For each epoch
        for i in range(1,epochs+1):
            # Print fitness status
            # and log to file for later use if need be
            temp = pd.DataFrame(population)
            temp.insert(0, 'Fitt', fitness_vec)
            print temp
            print temp.describe(percentiles=[])
            temp.to_csv(self.log_path+'/population.csv', index=False)
            print "Epoch %3i time: %.2fmin"%(i,(time.time() - time_before)/60)
            print "="*120
            epoch_acc = epoch_acc.append({'Max':max(fitness_vec), 'mean':sum(fitness_vec)/len(fitness_vec), 'Min':min(fitness_vec)}, ignore_index=True)
            epoch_acc.to_csv(self.log_path+'/LifCyle.csv', index=False)
            time_before = time.time()
            # For each individual
            for j in range(0, population_size):
                # Mutate
                # 3 random vectors excluding current individual
                mutation_vectors = range(0, population_size)
                mutation_vectors.remove(j)
                candidate = random.sample(mutation_vectors, 3)

                # Diff of 2 random vectors
                difference = [population[candidate[0]][k] - population[candidate[1]][k] for k in range(len(bounds))]

                # mutation_factor*diff + third random vector
                donor = [population[candidate[2]][k] + mutation_factor*difference[k] for k in range(len(bounds))]

                # Round and bound donor
                donor = self.validate_individual(donor, bounds, float_index_list)

                # Crossover
                recombined = []
                for k in range(len(bounds)):
                    if(random.random() <= crossover_factor):
                        recombined.append(donor[k])
                    else:
                        recombined.append(population[j][k])
                recombined = self.validate_individual(recombined, bounds, float_index_list)
                print [round(p, 2) for p in recombined],
                # Selection: Greedy
                # if new individual is better than the current, replace it
                Title =  "Epoch: %3i, Individual: %3i - "%(i,j) + str([round(p, 2) for p in recombined])
                recombined_fit = self.fitness(recombined, training_epochs, title=Title)
                print round(recombined_fit, 2), 
                if recombined_fit > fitness_vec[j]:
                    fitness_vec[j] = recombined_fit
                    population[j] = recombined
                    print " Survived."
                else:
                    print " Died."
        
        # Print fitness status
        # and log to file for later use if need be
        temp = pd.DataFrame(population)
        temp.insert(0, 'Fitt', fitness_vec)
        print temp
        print temp.describe(percentiles=[])
        temp.to_csv(self.log_path+'/population.csv', index=False)
        print (time.time() - time_before)/60
        print "="*120
        time_before = time.time()
    

# evolutivo para optimizar erro/razao entre erro de treino e erro de validacao; quando os dois divergem e sinal de overfitting
# tensorboard --logdir logs/1
def main():
# How to load data
    handler = neuralNet()
    handler.Load_Data("Dataset.txt")
    handler.Preprocess()
    handler.Split()
# How to train a modelx
    # handler.model_dense()
    # handler.fit_model(plot=True)
# how to optimize a model with diferential evolutionary algoritm
    #         batch     learn_rate   opti   [dropout nodes    activ]*10
    bounds = [(10,100), (0.001, 1), (1,7), (0, 0.3), (5, 35), (0, 10), (0, 0.3), (5, 35), (0, 10), (0, 0.3), (5, 35), (0, 10), (0, 0.3), (5, 35), (0, 10), (0, 0.3), (5, 35), (0, 10)]
    # indexes of which positions in bounds should NOT be integers
    float_indexes = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    handler.differential_evolution(4, 0.8, 0.6, bounds, float_indexes, 50, training_epochs = 2, file='logs/population.csv')
if __name__ == '__main__':
       main()
        