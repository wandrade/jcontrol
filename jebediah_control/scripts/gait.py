#!/usr/bin/env python
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from jcontroller import jcontroller
import pymatlab
import time
import numpy as np
from math import *
from random import *

#def main():
    # initiate robot
j = jcontroller(simulation=True)
    # Start matlab session
matSess=pymatlab.session_factory()
matSess.run("addpath(genpath('~/catkin_ws/src'))")

def joinLists(a, b, steps = 30):
    # generate intermediary stepts
    c = []
    loa = [] # list of actuators
    ini = a[-1]
    end = b[0]
    for i in range (4): # for each leg
        for j in range(3): # for each motor
            step =  (end[i][j]-ini[i][j])/steps
            if step == 0:
                loa.append([ini[i][j]] * steps)
            else:
                loa.append(np.arange(ini[i][j], end[i][j], step).tolist())
    for i in range(steps):
        c.append([
            [loa[0][i], loa[1][i], loa[2][i]],
            [loa[3][i], loa[4][i], loa[5][i]],
            [loa[6][i], loa[7][i], loa[8][i]],
            [loa[9][i], loa[10][i], loa[11][i]]
        ])
    # return a + c + b
    return c + b

def run():
    j.set_joints([0,0,0,0,0,0,0,0,0,0,0,0], mode='deg')
    f = open('Dataset_example.txt', 'w')
    for iterations in range(4):
        currentgait = []
        angList = []
        actions = []
        states = []
        # Generate path
        # (direction, walkDistance, bh, plot, steps)
        while True:
            if randint(0,1):
                cmd = "ang = GenerateAngularGait(%d, %f, 0.13, 0, %d)"%(0, 0.1, 350)
            else:
                cmd =        "ang = GenerateGait(%d, %f, 0.13, 0, %d)"%(0, 0.1, 350)
            try:
                print matSess.run(cmd)
                print iterations, ": " , cmd
                break
            except:
                pass
        currentgait = matSess.getvalue('ang').tolist()
        # get current position
        currentpos = j.get_state()
        currentpos = currentpos.Angles
        currentpos = [degrees(c) for c in currentpos]
        currentpos = [list(currentpos[0:3]),list(currentpos[3:6]),list(currentpos[6:9]),list(currentpos[9:12])]
        currentpos = [currentpos, currentpos]
        # generate transition gait
        angList = joinLists(currentpos, currentgait, steps = 30)
        # Convert to vector of floats
        for S in angList:
            step = []
            for l in S:
                for k in l:
                    step.append(k)
            actions.append(step)

        # Run loop
        for step in actions:
            # get state
            states.append(j.get_state())
            # print euler_from_quaternion ([states[-1].Orientation.x, states[-1].Orientation.y, states[-1].Orientation.z, states[-1].Orientation.w])
            # set action
            j.set_joints(step, mode='deg')
            # wait period
            time.sleep(0.05)
        # calculate velocity
        xVel = round(np.mean([s.Twist.linear.x for s in states][31:-1]),3)
        yVel = round(np.mean([s.Twist.linear.y for s in states][31:-1]),3)
        aVel = round(np.mean([s.IMU.angular_velocity.z for s in states][31:-1]),3)
        # write to file
        #header
        print ""
        print ("x_vel_set y_vel_set angular_vel_set motor_state_0 motor_state_1 motor_state_2 motor_state_3 motor_state_4 motor_state_5 motor_state_6 motor_state_7 motor_state_8 motor_state_9 motor_state_10 motor_state_11 ground_colision_0 ground_colision_1 ground_colision_2 ground_colision_3 orientation_quaternion_x orientation_quaternion_y orientation_quaternion_z orientation_quaternion_w angular_vel_x angular_vel_y angular_vel_z linear_acceleration_x linear_acceleration_y linear_acceleration_z linear_velocity_x linear_velocity_y linear_velocity_z action_0 action_1 action_2 action_3 action_4 action_5 action_6 action_7 action_8 action_9 action_10 action_11")
        print ""
        for i in range(len(actions)):
            ### INPUTS
            # setpoints
            line = "%f %f %f "%(xVel, yVel, aVel)
            # motors States
            line = line + "%f %f %f %f %f %f %f %f %f %f %f %f "%(states[i].Angles[0], states[i].Angles[1], states[i].Angles[2], states[i].Angles[3], states[i].Angles[4], states[i].Angles[5], states[i].Angles[6], states[i].Angles[7], states[i].Angles[8], states[i].Angles[9], states[i].Angles[10], states[i].Angles[11])
            # collision
            line = line + "%d %d %d %d "%(states[i].Ground_Collision[0], states[i].Ground_Collision[1], states[i].Ground_Collision[2], states[i].Ground_Collision[3])
            # IMU - mag - orientation
            line = line + "%f %f %f %f "%(states[i].IMU.orientation.x, states[i].IMU.orientation.y, states[i].IMU.orientation.z, states[i].IMU.orientation.w)
            # IMU - gyro
            line = line + "%f %f %f "%(states[i].IMU.angular_velocity.x, states[i].IMU.angular_velocity.y, states[i].IMU.angular_velocity.z)
            # IMU - accel
            line = line + "%f %f %f "%(states[i].IMU.linear_acceleration.x, states[i].IMU.linear_acceleration.y, states[i].IMU.linear_acceleration.z)
            # velocities
            line = line + "%f %f %f "%(states[i].Twist.linear.x, states[i].Twist.linear.y, states[i].Twist.angular.z)
            ### OUTPUTS
            # Action
            line = line + "%f %f %f %f %f %f %f %f %f %f %f %f"%(actions[i][0], actions[i][1], actions[i][2], actions[i][3], actions[i][4], actions[i][5], actions[i][6], actions[i][7], actions[i][8], actions[i][9], actions[i][10], actions[i][11])
            ### WRITE
            f.write('%s\n' % line)
        print xVel, yVel, aVel;
        
run()


