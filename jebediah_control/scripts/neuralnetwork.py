#! /usr/bin/env python
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.models import model_from_json
from math import sqrt, isnan
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

import gc
import time
import os
import errno
import random
import pandas as pd
import numpy as np 
import pickle
from fourierseries import *
from memory_profiler import profile

def plot_corr_ellipses(data, ax=None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)
    # 2222
    
    return ec

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def amap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class neuralNet(object):
    def __init__(self, *args):
        pd.options.display.max_rows = 12
        pd.options.display.float_format = '{:.2f}'.format
        self.log_path = os.path.dirname(os.path.realpath(__file__))
        self.log_path = self.log_path + "/model"
        mkdir_p(self.log_path)
        mkdir_p(self.log_path+"/Evolutionary_logs")
        mkdir_p(self.log_path+"/Tensor_logs")
        random.seed(1)
        np.random.seed(1)
    
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
        # print "%15s"%series.name, " mapped: From", min_val, "-", max_val, " to ", min_scale, "-", max_scale
        scale = (max_val - min_val) / (max_scale - min_scale)
        return series.apply(lambda x:((x - min_val) / scale) - (max_scale - min_scale)/2)

    def Preprocess(self):
        """Preprocess and syntesize data for neural net
        """
        df = self.data
        # x_vel_set y_vel_set angular_vel_set motor_state_0 motor_state_1 motor_state_2 motor_state_3 motor_state_4 motor_state_5 motor_state_6 motor_state_7 motor_state_8 motor_state_9 motor_state_10 motor_state_11 ground_colision_0 ground_colision_1 ground_colision_2 ground_colision_3 orientation_quaternion_x orientation_quaternion_y orientation_quaternion_z orientation_quaternion_w angular_vel_x angular_vel_y angular_vel_z linear_acceleration_x linear_acceleration_y linear_acceleration_z linear_velocity_x linear_velocity_y linear_velocity_z action_0 action_1 action_2 action_3 action_4 action_5 action_6 action_7 action_8 action_9 action_10 action_11
     # SINTETIC DATA
            # Inputs will be:
                # Setpoints
                # Current error
                # Robot state
                # Current velocities
                # Last actions

        # error module vector, where the vector is a 3D vecotr of x an y and angular velocities
        x_err = df["x_vel_set"]-df["linear_velocity_x"]
        y_err = df["y_vel_set"]-df["linear_velocity_y"]
        w_err = df["angular_vel_set"]-df["angular_vel_z"]
        df["error"] = np.sqrt(x_err**2 + y_err**2 + w_err**2)
        
        # put error column after setpoints
        cols = df.columns.tolist()
        cols = cols[:3] + [cols[-1]] + cols[3:-1]
        df = df[cols]

        # last actions is simply a delayed action vector
        for i in range(12):
            name = "action_%i"%i    
            delayed_name = "act_delayed_%i"%i
            df[delayed_name] = df[name].shift(1)
        df.fillna(0, inplace=True)

        # put actions lats
        cols = df.columns.tolist()
        cols = cols[:-24] + cols[-12:] + cols[-24:-12]
        df = df[cols]
        
     # NORMALIZATION
        for i in range(12):
            name = "motor_state_%i"%i
            df[name] = self.linear_scale(df[name], min_val=-90, max_val=90)
            name = "act_delayed_%i"%i
            df[name] = self.linear_scale(df[name], min_val=-90, max_val=90)
        # the other values are already small numbers

     # FOURIER
        # calculate actions output as fourrier transform and put it on the df
        df_list = get_chunk_list(df)
        # number of frequencies to aproximate fourrier series
        N = 20
        for k,d in enumerate(df_list): # for each chunk of df
            n = d.shape[0]
            image = []
            for i in range(12): # for each motor action in that chunk
                name = "action_%i"%i
                t = np.linspace(0, n*0.02, n, endpoint=False)
                # For each action, calculate its fourrier series and period
                T = get_period(d[name].values, t)
                S, W = fourier_series_coeff(np.array([d[name].values,t]), T, N)
                # Add values to df
                array = ft_to_array(S, T, norm=True)
                image.append(array)
                for j, f in enumerate(array):
                    d["act_%d_c%i"%(i,j)] = [f]*n
                # Reconstruct S
                # t0 = t
                # t = np.linspace(0, 2*n*0.02, 10*n, endpoint=False)
                # S, T = array_to_ft(d.iloc[0,-(N+3):].values, norm=True)
                # plot_fs(S,T,t, real = d[name].values)#, real_time = t0)
                # remove action column
                d.drop(name, axis=1, inplace=True)
            # Add output as 2D image
            # df['Action_2D'] = image
            # print "Chunk %3i/%3i processed."%(k+1,len(df_list))
            # rearange image to agrupate motors
            # image= np.array([
            #     image[0],
            #     image[3],
            #     image[6],
            #     image[9],
            #     image[1],
            #     image[4],
            #     image[7],
            #     image[10],
            #     image[2],
            #     image[5],
            #     image[8],
            #     image[11]
            # ])
            # # Show image
            # x_set = d["x_vel_set"].iloc[0]
            # y_set = d["y_vel_set"].iloc[0]
            # a_set = d["angular_vel_set"].iloc[0]
            # title = "Lin:(%.3f, %.3f) Ang: %.3f"%(x_set, y_set, a_set)
            # plt.title(title)
            # plt.xticks(np.arange(23),
            # ["T", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", 
            # "b0","b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"])

            plt.yticks(np.arange(12),[
                "leg_1_coxa",
                "leg_2_coxa",
                "leg_3_coxa",
                "leg_4_coxa",
                "leg_1_femur",
                "leg_2_femur",
                "leg_3_femur",
                "leg_4_femur",
                "leg_1_tibia",
                "leg_2_tibia",
                "leg_3_tibia",
                "leg_4_tibia",
            ])
            plt.imshow(image)
            plt.show()
        
        # cleanup
        self.data.drop(self.data.columns, axis=1, inplace=True)
        df.drop(df.columns, axis=1, inplace=True)
        for d in df_list:
            d.drop(df.columns, axis=1, inplace=True)
            
        # for i, n in enumerate(self.data.columns):
        #     print i, n
        self.data = pd.concat(df_list)

    def Split(self, validation_proportion=0.30, target_numbers=516, randomize=False, delete_original=True):
        """Split the class data into targets and labels for a validation and training and return a dictonary containing all 4 of the sets
        also keep a copy from it into the class.
        
        Keyword Arguments:
            validation_proportion {float} -- [Which proportion of the data will me separated for validation] (default: {0.3})
            target_numbers {int} -- [Number of targets (must all be the end of the dataset)] (default: {12})
            randomize {bool} -- [If false get data from end of vector, if true get random portion]
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
        
        if randomize is True:
            # get position where to starts cutting
            random_portion = np.random.randint(0,len(self.data)-validation_length-1) # dont get tips for ease of implementation
            # Remove validation of dataset and save to training
            training = self.data.drop(self.data.index[random_portion:validation_length+random_portion])
            validation = self.data.drop(training.index)
        else:
            training = self.data.head(training_length) 
            validation = self.data.tail(validation_length)
        
        # Separate target and validation
        column_numbers = len(validation.columns) # keep it variable due to possible sintetic labels
        tra_labels = training.iloc[:,0:column_numbers-target_numbers]
        tra_targets = training.iloc[:,column_numbers-target_numbers:]
        val_labels = validation.iloc[:,0:column_numbers-target_numbers]
        val_targets = validation.iloc[:,column_numbers-target_numbers:]

        if delete_original:
            self.data.drop(self.data.columns, axis=1, inplace=True)

        self.dataset = {'training':{'labels':tra_labels, 'targets':tra_targets}, 'validation':{'labels':val_labels, 'targets':val_targets}}.copy()
        return self.dataset
    
    def set_model(self, batch = 50, learning_rate=0.01, optimizer = 2, topology = [[0, 24, 1, 1],[0, 24, 0, 1],[0.1, 24, 1, 1],[0, 24, 1, 1],[0, 12, 1, 1]]):
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
        self.optimizer = optimizer
        self.learning_rate = learning_rate
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
    
        self.model_name = "NN"
        input = len(self.dataset['training']['labels'].columns)
        n = len(topology)
        # [dropout layer_nodes activation layer_type]
        model = Sequential()
        for i, layer in enumerate(topology):
                if i == n-1: layer[1] = len(self.dataset['training']['targets'].columns)
                if layer[3] == 0: # Dense layer
                    model.add(Dense(layer[1], input_dim=input, activation=layer[2])) 
                # elif layer[3] == 1: # Convolutional layer
                #     model.add(Conv1D(layer[1], 2, activation=layer[]))
                # elif layer[3] == 2: # LTSM layer
                #     model.add(LSTM(layer[1], input_dim=input, activation=layer[2])) 
                
                model.add(Dropout(layer[0]))
                input = layer[1]
        model.compile(loss='mean_squared_error', optimizer=opti, metrics=['mse'])
        self.model = model
    
    def fit_model(self, epochs=5, plot=None, verbose=0, log=True, title="Model Fitting", clean=False):
        
        if log == True: # create log path
            log_path =  self.log_path + "/Tensor_logs/" + time.strftime('%Y-%m-%d_%H-%M-%S')
            print "Logging to ", log_path
            mkdir_p(log_path)
            tensorboard = TensorBoard(log_dir=log_path)
            # log for tensorboard visualization
            plot_model(self.model, to_file=log_path+'/model.png', show_shapes=True, show_layer_names=True)
            cb = [tensorboard]
        else:
            cb = []
        # fit model
        history = self.model.fit(
                        self.dataset['training']['labels'].values, 
                        self.dataset['training']['targets'].values, 
                        epochs=epochs, 
                        batch_size=self.batch, 
                        callbacks=cb,
                        validation_data=(
                            self.dataset['validation']['labels'].values, 
                            self.dataset['validation']['targets'].values),
                        verbose = verbose)
        if plot is not None:
            plt.clf()
            plt.suptitle(title)

            # summarize history for loss
            plt.subplot(1,1,1)
            plt.plot(history.history['loss'])
            if 'val_loss' in history.history.columns:
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
        if isnan(history.history['loss'][-1]) or isnan(history.history['val_loss'][-1]) or isnan(history.history['loss'][-1]):
            loss_tra =  float(0.0)
            loss_val =  float(0.0)
        loss_val = float(history.history['val_loss'][-1])
        loss_tra = float(history.history['loss'][-1])

        # clean
        if clean:
            del history
            del self.model
            K.clear_session()
            gc.collect()        
        return loss_val, loss_tra
    
    def save_model(self, path=None):
        if path is None: path = self.log_path
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path+"/model_topology.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path+"/model_weights.h5")
        # save model's optimizer
        with open(path+"/model_optimizer.bin", "wb") as opti_file:
            pickle.dump([self.optimizer, self.learning_rate], opti_file)
        print("Saved model to disk")

    def load_model(self, path=None):
        if path is None: path = self.log_path
        # load json and create model
        json_file = open(path+"/model_topology.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(path+"/model_weights.h5")
        # Load model optimizer
        opti_file = open(path+"/model_optimizer.bin", 'rb')
        self.optimizer, self.learning_rate = pickle.load(opti_file)
        # Optimizer
        if self.optimizer == 1:
            opti = optimizers.Adam(lr=self.learning_rate, epsilon=None, decay=0.0)
        elif self.optimizer == 2:
            opti = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        elif self.optimizer == 3:
            opti = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)
        elif self.optimizer == 4:
            opti = optimizers.Adagrad(lr=self.learning_rate, epsilon=None, decay=0.0)
        elif self.optimizer == 5:
            opti = optimizers.Adadelta(lr=self.learning_rate, rho=0.95, epsilon=None, decay=0.0)
        elif self.optimizer == 6:
            opti = optimizers.Adamax(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        elif self.optimizer == 7:
            opti = optimizers.Nadam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # compile model
        self.model.compile(loss='mean_squared_error', optimizer=opti, metrics=['mse'])
        self.model.predict(np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))
        print("Loaded model from disk")

    def model_eval(self):
        # evaluate the model
        scores = self.model.evaluate(self.dataset['validation']['labels'],self.dataset['validation']['targets'] , verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]))
        return scores[1]
    
    def predict(self, input):
        """Predict next step
        
        Arguments:
            input {list} -- Input vector with all model inputs
        
        Returns:
            list -- List with next steps
        """
        return self.model.predict(input)
    
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
    
    def print_model(self, b, l, opti, top):

            # Optimizer
        if opti == 1: opti_name = "Adam"
        elif opti == 2: opti_name = "SGD"
        elif opti == 3: opti_name = "RMSprop"
        elif opti == 4: opti_name = "Adagrad"
        elif opti == 5: opti_name = "Adadelta"
        elif opti == 6: opti_name = "Adamax"
        elif opti == 7: opti_name = "Nadam"
        
        print "   Batch size........: %d"%b
        print "   Learning rate.....: %.3f"%l
        print "   Optimizer.........: %s"%opti_name
        print "Layers:"
        j = 0
        for i, layer in enumerate(top):
            # Get Activation function
            if layer[2] == 0: 
                j = j + 1
            else:
                if layer[2] == 1: activation = 'relu'
                elif layer[2] == 2: activation = 'softmax'
                elif layer[2] == 3: activation = 'elu'
                elif layer[2] == 4: activation = 'selu'
                elif layer[2] == 5: activation = 'softplus'
                elif layer[2] == 6: activation = 'softsign'
                elif layer[2] == 7: activation = 'tanh'
                elif layer[2] == 8: activation = 'sigmoid'
                elif layer[2] == 9: activation = 'hard_sigmoid'
                elif layer[2] == 10: activation = 'linear'
                
                if layer[3] == 0: layer_type = "Dense"
                elif layer[3] == 1: layer_type = "LSTM"
                elif layer[3] == 2: layer_type = "Conv1D"

                print "      %2d. dropout: %1.3f  Nodes: %3d  Activation: %12s  Type: %6s"%(i+1-j, layer[0], layer[1], activation, layer_type)

    def convert_to_model(self, vector):
        batch = vector[0]
        lr = vector[1]
        opti = vector[2]
        topology = []
        layer_num = (len(vector)-3)/4
        for i in range(layer_num):
            topology.append([vector[3+i*4], vector[3+i*4+1], vector[3+i*4+2], vector[3+i*4+3]])
            i = i + 4
        return batch, lr, opti, topology
    
    def fitness(self, individual, training_epochs, verbosity=0, title="Individual Fitness", v=False):
        if v: print 'Fitness eval: ', training_epochs,"epochs"
        b, l, opti, top = self.convert_to_model(individual)
        if v: self.print_model(b, l, opti, top)
        alpha = 0.7
        # create model
        self.set_model(b, l, opti, top)
        loss_val, loss_tra = self.fit_model(epochs=training_epochs, log=False, verbose=verbosity, plot=1, title=title, clean=True)
        
        #fitness = (1-alpha)*loss_val + alpha*abs(loss_val - loss_tra)
        #print "Fitness: %.2f    Loss parcel: %.2f    Diff parcel: %.2f"%(fitness, (1-alpha)*loss_val, alpha*abs(loss_val - loss_tra))
        fitness = alpha*loss_val + (1-alpha)*loss_tra
        # train and evaluate
        return fitness

    @profile
    def differential_evolution(self, population_size, mutation_factor, crossover_factor, bounds, float_index_list, epochs, training_epochs=100, file=None):
        ### INITIALIZE POPULATION
        population = []
        fitness_vec = []
        epoch_fitness = pd.DataFrame(columns=['Max', 'mean', 'Min'])
        time_before = time.time()
        print "Initializing population."
        if file is None:
            for i in range(0, population_size):
                print "%2d/%2d"%(i+1,population_size)
                indv = []
                for j in range(len(bounds)):
                    indv.append(random.uniform(bounds[j][0],bounds[j][1]))
                indv = self.validate_individual(indv, bounds, float_index_list)
                population.append(indv)
                # calculate fitness
                fitness_vec.append(self.fitness(indv, training_epochs, verbosity=1, v=True))
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
            temp.to_csv(self.log_path+'/Evolutionary_logs/population.csv', index=False)
            print "Epoch %3i time: %.2fmin"%(i,(time.time() - time_before)/60)
            print "="*120
            epoch_fitness = epoch_fitness.append({'Max':max(fitness_vec), 'mean':sum(fitness_vec)/len(fitness_vec), 'Min':min(fitness_vec)}, ignore_index=True)
            epoch_fitness.to_csv(self.log_path+'/Evolutionary_logs/LifeCycle.csv', index=False)
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
                
                # minimize fitness
                if recombined_fit < fitness_vec[j]:
                    fitness_vec[j] = recombined_fit
                    population[j] = recombined
                    print " Spawn survived."
                else:
                    print " Spawn died."
        
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
    
    @profile
    def plot_lifecyle(self, path=None):
        if path is None:
            lcl = pd.read_csv(self.log_path+'/Evolutionary_logs/LifeCycle.csv')
            pop = pd.read_csv(self.log_path+"/Evolutionary_logs/population.csv")
        else:
            lcl = pd.read_csv(path+"/LifeCycle.csv")
            pop = pd.read_csv(path+"/population.csv")
        fig = plt.figure()
        plt.subplot2grid((2,2),(0, 0))
        ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
        ax2 = plt.subplot2grid((2,2), (1,0), colspan=1)
        ax3 = plt.subplot2grid((2,2), (1,1), colspan=1)
        
        # Fitnss over time
        ax1.set_title("Population fitness on validation dataset")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Fitness")
        ax1.plot(lcl['Max'])
        ax1.plot(lcl['mean'])
        ax1.plot(lcl['Min'])
        ax1.legend(["Max","Mean", "Min"])
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax1.grid()

        # Correlation
        ax2.set_title("Correlation")
        corr_df = pop.copy()
        corr_df = pd.DataFrame({'Fitness':corr_df['Fitt'], 'Batch':corr_df['0'],'L.Rate':corr_df['1'], 'Optimizer':corr_df['2'],'Dropout':corr_df[['3', '6', '9', '12', '15']].mean(axis=1),'Nodes per layer':corr_df[['4', '7', '10', '13', '16']].mean(axis=1), 'Activation':corr_df[['5', '8', '11', '14', '17']].mean(axis=1)})
        corr = corr_df.corr() 
        m = plot_corr_ellipses(corr, ax=ax2)
        cb = fig.colorbar(m,ax=ax2)
        cb.set_label('Correlation coefficient')
        ax2.margins(0.1)
        
        # Boxplot
        df_norm = corr_df.copy()
        # Normalize
        df_norm['Activation'] = amap(df_norm['Activation'], 0, 10, 0, 1)
        df_norm['Batch'] = amap(df_norm['Batch'], 0, 100, 0, 1)
        df_norm['Dropout'] = amap(df_norm['Dropout'], 0, 0.5, 0, 1)
        df_norm['L.Rate'] = amap(df_norm['L.Rate'], 0.0001, 0.1, 0, 1)
        df_norm['Nodes per layer'] = amap(df_norm['Nodes per layer'], 5, 35, 0, 1)
        df_norm['Optimizer'] = amap(df_norm['Optimizer'], 1, 7, 0, 1)
        df_norm.boxplot(vert=0)
        # scatter
        print corr_df
        print df_norm
        for i in range(1, 7+1):
            y = df_norm[df_norm.columns[i-1]].tolist()
            # Add some random "jitter" to the x-axis
            x = np.random.normal(i, 0.04, size=len(y))
            ax3.plot(y, x, 'r.', alpha=0.5)

        ax3.set_title('Final population dispersion')
        ax3.set_xlim([-0.12, 1.12])
        ax3.set_xticklabels(['0%', "25%", "50%", "75%", "100%"])
        ax3.set_xticks([0, 0.25, 0.50, 0.75, 1])
        ax3.set_xlabel("Search range")
        plt.show()

# tensorboard --logdir logs/1
def main():
    # How to load data
    # first you need to load the data to optimize then train the model
    # there is a smaller dataset for evolutionary puproses caled Dataset_evo.txt instide neuralnet/logs folder
    # and a second bigger one called Dataset which is the real training dataset, load each one accordinly
    start_time = time.time()
    print "Started at %s"%time.strftime('_%Y-%m-%d_%H-%M-%S')
    handler = neuralNet()
    handler.log_path = "/home/werner/catkin_ws/src/jcontrol/jebediah_control/scripts/model"
    handler.Load_Data(handler.log_path+"/Datasets/Dataset_evo.txt")
    handler.Preprocess()
    handler.Split(delete_original=False)
    # how to optimize a model with diferential evolutionary algoritm
    # if yo have no model, you first have to run the optimizer to determine the best topology and hiperparameters
    # run the code bellow, where bounds specify the search range and the max number for instance 5 layers
    #         batch    learn_rate  opti   [dropout  nodes      activ    type]  [dropout  nodes      activ    type]  [dropout  nodes      activ    type]  [dropout  nodes      activ    type]  [dropout  nodes      activ    type]  [dropout  nodes      activ    type] ... You can use as many as ou want as long as each layer has those 4 parameter   
    #bounds = [(5,50), (0.0001, 1), (1,7), (0, 0.5), (45, 600), (0, 10), (0,0), (0, 0.5), (45, 600), (0, 10), (0,0), (0, 0.5), (45, 600), (0, 10), (0,0), (0, 0.5), (45, 600), (0, 10), (0,0), (0, 0.5), (45, 600), (0, 10), (0,0), (0, 0.5), (45, 600), (0, 10), (0,0)]
    # some of the values in 'bound' have to be rounded since they are ony flags, so we have to pass a list of the lumbers that should not be rounded as bellow
    # indexes of which positions in bounds should NOT be integers
    #float_indexes = [1, 3, 7, 11, 15, 19, 23]
    # this is the actual algoritm call, it logs a 'population' file every iteration so if the training stops you can restart it by passing the file as argument
    # if you dont want to continue from where it stoped, just remove the 'file' argument]
    # in this file you can also se  your last population and use it as you will
    # Header: differential_evolution( population_size, mutation_factor, crossover_factor, bounds, float_index_list, epochs, training_epochs=100, file=None):            
    #handler.differential_evolution(15, 0.3, 0.4, bounds, float_indexes, 20, training_epochs = 10)#, file='logs/population.csv')
    # this plot the file LifeCycle of the evolutionary algorithm, this files is incremented at each generation even though the training is stoped for some reason
    #handler.plot_lifecyle()

    # How to train a model
    # After having a good idea as to which model you should use, load the bigger dataset and run this fraction of code to train and save the model on a file
    # got this from optimization
    model_params = [30,0.6, 1, 0.01, 520, 9, 0, 0.01, 520, 9, 0, 0.01, 520, 9, 0, 0.01, 512, 9, 0]
    # Convert population string to parameters to create the model
    batch, lr, opti, topo = handler.convert_to_model(model_params)
    handler.print_model(batch, lr, opti, topo)
    # Create model
    handler.set_model(batch=batch, learning_rate=lr, optimizer=opti, topology=topo)
    # Here we will use a k-fold validation teqnique to trains and validade our model k times
    # Vector to keep fitness over each iternation
    fitness=[]
    mse=[]
    for k in range (10):
        st = time.time()
        print '='*100
        # resplit model randomly (instead of using only the end part of the data as above)
        handler.Split(validation_proportion=0.30, randomize=True, delete_original=False)
        plot_title = "Model Training: %i"%k
        # train, plot and save value
        fitness.append(handler.fit_model(epochs=5, plot=1, verbose=1, log=True, title=plot_title))
        print ""
        print 'Model validation:'
        print 'Fitness:           ',fitness[-1]
        mse.append(handler.model_eval())
        time_spent = time.time()-st
        print time_spent/(60*60)
    print '-'*100
    print "K-fold validation results:"

    print 'Fitness (%):'
    fitness = pd.Series(fitness)
    print fitness
    print fitness.describe(percentiles=[])

    print 'Mean Absolute Error (%):'
    mse = pd.Series(mse)
    print mse
    print mse.describe(percentiles=[])

    print "Final error: %.1f +- %.1f"%(np.sqrt(mse.mean()), sqrt(mse.std()))
    # train final model with all data
    print '-'*100
    handler.Split(validation_proportion=0.001)
    fitness = handler.fit_model(epochs=10, plot=2, log=True, title='Model training', verbose=2)
    print 'Fitness: %.2f%%'%fitness
    # Evaluate model
    handler.model_eval()
    # save model to file:
    handler.save_model()

    end_time = time.time()
    print ""
    print "Finished at %s"%time.strftime('_%Y-%m-%d_%H-%M-%S')
    total = end_time - start_time
    print "Time consumption: %.2fh"%float(total/(60*60))

if __name__ == '__main__':
       main()
        
