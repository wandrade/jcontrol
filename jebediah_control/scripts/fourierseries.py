#! /usr/bin/env python
import os

import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy import fftpack
import time
import pandas as pd
from math import cos, sin

def timeit(method):
    """Time decorator, this can be used to measure the function elapesd time without interfering on its funcionality
    To use it, put the following decorator befor any function:
    @timeit
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.3f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed

def interpolate(t, pairs):
    if type(t) is np.ndarray or type(t) is list:
        return_list = []
        for T in t:
            return_list.append(interpolate(T, pairs))
        return np.array(return_list) 
    # Calculate distance from t to each element in time vector
    distance = abs(pairs[1] - t)
    # get the next value
    idx = np.sort([np.argpartition(distance, 2)[:2]])[0]
    d = distance[idx]
    d_range = d[0]+d[1]
    d = d_range-d
    weights = d/d_range
    # calculate the avg of those points based on distance
    closest = np.average(pairs[0][idx], weights=weights, axis=-1)
    try:
        return closest[0]
    except Exception as e:
        return closest

def ft_to_array(ft, T):
    #separate imag from real
    series = np.concatenate((ft.real, ft.imag))
    return np.concatenate((np.array([T]), series))

def array_to_ft(array):
    T = array[0]
    real = array[1:1+len(array)/2]
    imag = array[1+len(array)/2:]
    S = real + imag*1j
    return S, T    

def plot_fs(S, T, t, real=None, real_time=None):
    fig, ax = plt.subplots()
    s_r = eval_rfft(S, t, T)
    ax.plot(t, s_r, 'b', label="Reconstruction")
    if real is not None and real_time is None:
        ax.plot(t, real, 'r', dashes=[4,1,1,1], label="Original")
    elif real is not None and real_time is not None:
        ax.plot(real_time, real, 'r', dashes=[4,1,1,1], label="Original")
    ax.legend()
    ax.grid()
    plt.show()

def fourier_series_coeff(f, T, N, return_complex=True):    
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function array
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.


    from: https://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
    """
    t = np.linspace(0, T, N, endpoint=False)
    y = np.fft.rfft(interpolate(t, f)) / t.size
    w = fftpack.rfftfreq(len(y))
    if return_complex:
        return y, w
    else:
        y *= 2
        return y[0].real, np.transpose(y[1:].real), np.transpose(-y[1:].imag)

def eval_rfft(S, time, T):
    N = 2*(len(S)-1)
    period = np.linspace(0, T, N)
    s = np.fft.irfft(S)*period.size
    s = s.reshape(len(s))
    if type(time) is list or type(time) is np.ndarray:
        vect = []    
        # make a constant time vector loop trough the limited period vector
        # generate ocilating time vector
        for t in time:
            val = t - int(t/T)*T
            vect.append(val)
        # evaluate
        aprox = interpolate(vect, np.array([s,period]))
        return aprox
    else:
        val = time - int(time/T)*T
        return interpolate(val, np.array([s,period]))

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if abs(x) <= 0.01 else x for x in values]

def get_period(s, t):
    max_val = max(s)
    # get times when maximum occurs
    max_occur = t[(s == max_val)]
    if len(max_occur) > 2:
        T = max_occur[1] - max_occur[0]
        return T
    else:
        # If could find a precise period, return max t
        return max(t)

def process_batch(df):
    
    steps = [
        df["action_9"].tolist(),  
        df["action_10"].tolist(), 
        df["action_11"].tolist()
    ]

    period = 0.020
    ax_time = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    ax_fourier = []
    ax_fourier.append(plt.subplot2grid((4, 3), (1, 0)))
    ax_fourier.append(plt.subplot2grid((4, 3), (1, 1)))
    ax_fourier.append(plt.subplot2grid((4, 3), (1, 2)))
    inverse_fourier = plt.subplot2grid((4, 3), (2, 0), colspan=3, rowspan=2)

    N = 20 # number of cosines for aproximation (real + imag)

    for i, s in enumerate(steps):
    # convert signal to time
        s = np.array([float(x) for x in s])
        t = np.linspace(0, len(s)*period, len(s), endpoint=False)
        # t = np.linspace(0, 2, 0, endpoint=False)
        # s = np.cos(2*np.pi*t)
        ax_time.plot(t, s)
    # Calculate period (only needed for the first iteration)
        if i == 0: 
            T = get_period(s,t)
        print T
    
    # Calculate N fourier series coeficients
        S, W = fourier_series_coeff(np.array([s,t]), T, N, return_complex=True)
        # plot_fs(S, T, t, real=s)
        ax_fourier[i].stem(W,S.real, label="real")
        ax_fourier[i].stem(W,S.imag, 'r', markerfmt='ro', linefmt='r--', label="imag")
        ax_fourier[i].grid()

    # Reconstruct the signal using fourier series and plot
        inverse_fourier.plot(t,s, 'b')


        S_array = ft_to_array(S, T)
        S, T = array_to_ft(S_array)
        s_r = eval_rfft(S, t, T)
        inverse_fourier.plot(t,s_r, 'r', dashes=[4,1,1,1])

    inverse_fourier.legend(['Original', 'Reconstructed'])
    inverse_fourier.set_title("Fourier series recnstruction")
        
    # ticks
    major_ticks = np.arange(-120, 121, 30)
    ax_time.set_yticks(major_ticks)
    inverse_fourier.set_yticks(major_ticks)

    ax_time.set_title('Time')
    ax_time.legend(["Coxa","Femur","Tibia"])
    ax_time.grid()
    # ax_time_reconstruct.legend(["Coxa","Femur","Tibia"])
    
    ax_fourier[0].set_title("Fourier components coxa")
    ax_fourier[1].set_title("Fourier components femur")
    ax_fourier[2].set_title("Fourier components tibia")
    inverse_fourier.grid()

    # plt.tight_layout()
    plt.show()
    
def get_chunk_list(df):
    """From a chunk of data with various setpoints, separates it in a list of smaller chunks 
    that each contain the setpoint transition steps and the setpoint steps
    
    Arguments:
        data {pd} -- All data read from file generated by gait.py
    """
    # create temporary columns with 1 shift
    df["tmp_1"] = df["x_vel_set"].shift(1).copy()
    df["tmp_2"] = df["y_vel_set"].shift(1).copy()
    df["tmp_3"] = df["angular_vel_set"].shift(1).copy()
    # create boolean column where everytime a value changes it gets a True
    df["diff_1"] = df["x_vel_set"] != df["tmp_1"]
    df["diff_2"] = df["y_vel_set"] != df["tmp_2"]
    df["diff_3"] = df["angular_vel_set"] != df["tmp_3"]
    # Remove auxiliary columns
    df.drop("tmp_1", axis=1, inplace=True)
    df.drop("tmp_2", axis=1, inplace=True)
    df.drop("tmp_3", axis=1, inplace=True)
    # Calculate final diff column
    df["diff"] = df["diff_1"] | df["diff_2"] | df["diff_3"]
    df.drop("diff_1", axis=1, inplace=True)
    df.drop("diff_2", axis=1, inplace=True)
    df.drop("diff_3", axis=1, inplace=True)
    df["groups"] = df["diff"].cumsum()
    df.drop("diff", axis=1, inplace=True)
    # Separate in groups and create a list of dataframes
    grouped = df.groupby("groups")
    groups = [group for _, group in grouped]
    # Separate each dataframe into other 2 dataframes, 1 with the 30 first steps and the other with the rest
    df_list = []
    for g in groups:
        df_list.append(g.iloc[:30])
        df_list.append(g.iloc[30:])
    # Remove uneeded data
    for d in df_list:
        d.drop("groups", axis=1, inplace=True)
    return df_list

def main(args):
    """ This script opens up a dataset file and analyses it's frequency spectrum.
        Since all legs do basicaly the same motions but with some delay between eachother, the script will only analyse one leg. 
        The goal is the look at the spectrum for diferent setpoints and try to determine what is the range and wich frequencies are more important on each motor of the leg.
    """
    # Setup
    pd.options.display.max_rows = 30
    pd.options.display.float_format = '{:.3f}'.format

    # File path
    print 'Loading data frame'
    log_path = os.path.dirname(os.path.realpath(__file__))
    log_path = log_path + "/model/Datasets/Dataset_evo.txt"
    dataset = pd.read_csv(log_path, sep=' ')

    print 'Separating chunks'
    dataset = get_chunk_list(dataset)
    print 'Analysing...'
    # loop
    for i, df in enumerate(dataset):
        print "-"*150
        print 'Set #%.3d'%i, 
        print 'Len: ', df.shape
        print df.iloc[0, :3]
        process_batch(df)
if __name__ == '__main__':
    args = 0
    main(args)