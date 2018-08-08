import os

import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy import fftpack
import time

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

def interpolate2(t, pairs):
    if type(t) is np.ndarray or type(t) is list:
        return_list = []
        for T in t:
            return_list.append(interpolate(T, pairs))
        return np.array(return_list) 
    # Calculate distance from t to each element in time vector
    distance = abs(pairs[1] - t)
    # get the 2 miminum distances and return its equialent of function
    try:
        return pairs[0][distance == min(distance)][0]
    except Exception as e:
        return pairs[0][distance == min(distance)]

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
    # print pairs
    closest = np.average(pairs[0][idx], weights=weights, axis=-1)
    try:
        return closest[0]
    except Exception as e:
        return closest

def fourier_series_coeff(f, T, N, return_complex=False):    
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
        return y[0].real, np.transpose(y[1:-1].real)[0], np.transpose(-y[1:-1].imag)[0]
@timeit
def eval_rfft(S, time, T, N):
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

def process_batch(steps, sp):
    period = 0.020
    print 'Batch: ', sp
    t = np.linspace(0, len(steps)*period, len(steps), endpoint=False)
    steps = zip(*steps)
    ax_time = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    ax_fourier = []
    ax_fourier.append(plt.subplot2grid((4, 3), (1, 0)))
    ax_fourier.append(plt.subplot2grid((4, 3), (1, 1)))
    ax_fourier.append(plt.subplot2grid((4, 3), (1, 2)))
    inverse_fourier = plt.subplot2grid((4, 3), (2, 0), colspan=3, rowspan=2)

    N = 2*20 # number of cosines for aproximation

    for i, s in enumerate(steps):
    # convert signal to time
        s = np.array([float(x) for x in s])
        # t = np.linspace(0, 2, 0, endpoint=False)
        # s = np.cos(2*np.pi*t)
        ax_time.plot(t, s)
    # Calculate period (only needed for the first iteration)
        if i == 0: 
            max_val = max(s)
            # get times when maximum occurs
            max_occur = t[(s == max_val)]
            if len(max_occur) > 2:
                T = max_occur[1] - max_occur[0]
            else:
                print 'error, could not find period'
                plt.show()
                break
        print T
    # Calculate N fourier series coeficients
        S, W = fourier_series_coeff(np.array([s,t]), T, N, return_complex=True)
        print len(S), len(W)
        print S, W
        ax_fourier[i].stem(W,S.real, label="real")
        ax_fourier[i].stem(W,S.imag, 'r', markerfmt='ro', linefmt='r--', label="imag")
        # ax_fourier[i].stem([0], [a0], 'k', markerfmt='ko', linefmt='k--', label = 'a0')
        # ax_fourier[i].stem(np.arange(1, 1 + N), a, 'r', markerfmt='ro', linefmt='r--', label = 'an')
        # ax_fourier[i].stem(np.arange(1, 1 + N), b, 'b', markerfmt='bo', linefmt='b-.',label = 'bn')
        ax_fourier[i].grid()

    # Reconstruct the signal using fourier series and plot
        inverse_fourier.plot(t,s, 'b')
        # s_i = np.fft.irfft(S)*np.linspace(0, T, N+2).size
        # inverse_fourier.plot(np.linspace(0,T,len(s_i)), s_i, 'g', dashes=[4,1,1,1])

        s_r = eval_rfft(S, t, T, N)
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
    

def main(args):
    size_l = []
    """ This script opens up a dataset file and analyses it's frequency spectrum.
        Since all legsdo basicaly the same motions but with some delay between eachother, the script will only analyse one leg. 
        The goal is the look at the spectrum for diferent setpoints and try to determine what is the range and wich frequencies are more important on each motor of the leg.
    """
    # File path
    log_path = os.path.dirname(os.path.realpath(__file__))
    log_path = log_path + "/model/Datasets/Dataset_evo.txt"

    # Open file
    with open(log_path, 'r') as f:
        for i in range(31):
            next(f)
        reader = csv.reader(f, delimiter =' ')
        dataset = list(reader)
    
    # loop
    set_point = dataset[0][0:3]
    batch = []
    for l in dataset:
        # Check if did not setpoint changed
        if l[0:3] == set_point:
            # If so add it to the list
            batch.append(l[-3:])
            # delete line to save memory
            del l
        else:
            # Process the current batch
            process_batch(batch, set_point)
            break
            # Erase it
            size_l.append(set_point)
            print size_l[-1]
            del batch
            batch = []
            # Start new batch
            set_point = l[0:3]
            # remove next 30 steps (transistion steps)
            for i in range(30):
                del dataset[0]
    print '--------'
    print min(size_l)

if __name__ == '__main__':
   main()