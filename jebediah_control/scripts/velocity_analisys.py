#!/usr/bin/env python

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from fourierseries import get_chunk_list, get_period, interpolate

def eval_period(s, T, time):
    N = len(s)
    period = np.linspace(0, T, N)
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


path = os.path.dirname(os.path.realpath(__file__))
full_steps = pd.read_csv(path + "/model/Datasets/Dataset_evo.txt", sep=" ")
processed_steps = pd.read_csv(path + "/model/Datasets/Dataset_processed.csv")


# Reference path
# get path batches
df_list = get_chunk_list(full_steps)
linear_steps = df_list[0]
linear_steps.drop(linear_steps.columns[0:32], axis=1, inplace=True)
n = linear_steps.shape[0]
t = np.linspace(0, n*0.02, n, endpoint=False)
T = get_period(linear_steps["action_0"].values, t)
# Drop repeated steps
linear_steps.drop(linear_steps.index[int(T/0.02):],axis=0, inplace=True)
for serie in linear_steps:
    linear_steps[serie] = np.radians(linear_steps[serie])
linear_steps.plot(grid=True, title="Linear loop")
linear_steps.to_csv(path + "/model/reference/forward.csv", index=False)

# do the same for angular movement
angular_steps = df_list[6]
angular_steps.drop(angular_steps.columns[0:32], axis=1, inplace=True)
angular_steps.reset_index(inplace=True)
n = angular_steps.shape[0]
t = np.linspace(0, n*0.02, n, endpoint=False)
T = get_period(angular_steps["action_0"].values, t)
# Drop repeated steps
angular_steps.drop(angular_steps.index[int(T/0.02)+1:],axis=0, inplace=True)
angular_steps.reset_index(inplace=True)
angular_steps.drop(["level_0", "index"],axis = 1, inplace = True)
for serie in angular_steps:
    angular_steps[serie] = np.radians(angular_steps[serie])
angular_steps.plot(grid=True, title="Angular loop")
angular_steps.to_csv(path + "/model/reference/rotate.csv", index=False)

# evaluate the file
vel_set_point = 0.8
T = exp_ang = np.exp(4.86364516) * np.exp(-2.15245784*vel_set_point)

vect = np.linspace(0, 100, 100/0.02, endpoint=False)
plt.figure()
for signal in angular_steps:
    aprox = eval_period(angular_steps[signal], T, vect)
    plt.plot(vect, aprox, label=signal)
plt.title("Evaluation")
plt.legend()
plt.grid()
plt.show()

# # Get periods
# # Drop all unwanted columns
# for j in range(12):
#     for i in range(1,23):
#         name = "act_%d_c%d"%(j,i)
#         processed_steps.drop(name, axis=1, inplace=True)
# processed_steps.drop(processed_steps.columns[6:-1], axis=1, inplace=True)
# processed_steps.drop("error", axis=1, inplace=True)
# processed_steps.drop("act_1_c0", axis=1, inplace=True)
# processed_steps.drop("act_11_c0", axis=1, inplace=True)

# processed_steps["lin"] = processed_steps[["x_vel_set", "y_vel_set"]].max(axis=1)
# processed_steps.drop("x_vel_set", axis=1, inplace=True)
# processed_steps.drop("y_vel_set", axis=1, inplace=True)
# processed_steps["act_0_c0"] = 10*processed_steps["act_0_c0"]

# linear = processed_steps.query("lin > angular_vel_set").copy()
# linear.drop("angular_vel_set", axis=1, inplace=True)
# linear = linear.query("lin > 0.05")
# angula = processed_steps.query("lin < angular_vel_set").copy()
# angula.drop("lin", axis=1, inplace=True)
# angula = angula.query("angular_vel_set > 0.2")

# # Calculate exponential aproximation
# # ang
# b = np.polyfit(angula["angular_vel_set"].values, np.log(angula["act_0_c0"].values), 1, w=np.sqrt(np.log(angula["act_0_c0"].values)))
# vel_ang = np.linspace(0.2,0.8,300)
# exp_ang = np.exp(b[1]) * np.exp(b[0]*vel_ang)
# print b
# # [-2.15245784  4.86364516]
# # lin
# b = np.polyfit(linear["lin"].values, np.log(linear["act_0_c0"].values), 1, w=np.sqrt(np.log(linear["act_0_c0"].values)))
# vel_lin = np.linspace(0.04,0.2,300)
# exp_lin = np.exp(b[1]) * np.exp(b[0]*vel_lin)
# print b
# # [-10.37872853   5.01919578]

# ax1 = plt.subplot(2,1,1)
# ax1.scatter(angula["angular_vel_set"], angula["act_0_c0"], label="Training")
# ax1.plot(vel_ang, exp_ang, 'r:', label="Exponetial fit")
# ax1.set_title("Angular vel")
# ax1.legend()

# ax2 = plt.subplot(2,1,2)
# ax2.scatter(linear["lin"], linear["act_0_c0"], label="Training")
# ax2.plot(vel_lin, exp_lin, 'r:', label="Exponetial fit")
# ax2.set_title("Linear vel")
# ax2.legend()

# plt.show()