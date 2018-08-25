#!/usr/bin/env python

import rospy
from jcontrol_msgs.msg import State, Action
from std_srvs.srv import Empty
import neuralnetwork
import time
from math import *
import numpy as np
from fourierseries import interpolate
import os
import pandas as pd
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


def amap(x, in_min, in_max, out_min, out_max):
    """Map a input 'x' from a range in_min-in_max to a range out_min_out_max
    
    Arguments:
        x {float} -- input value
        in_min {float} -- min value this input can have
        in_max {float} -- max value this input can have
        out_min {float} -- min value to map
        out_max {float} -- max value to map
    
    Returns:
        float -- Mapped value
    """
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class jcontroller:
    #             x    y  rot
    set_point = [0.0, 0.0, 0.1]
    action_publisher = None
    state = None
    prev_actions = [2]*12
    def __init__(self):
        rospy.init_node('jebediah_controler')
        # Create reset service
        rospy.loginfo("Waiting for gazebo services")
        rospy.wait_for_service('gazebo/reset_simulation')
        self.action_publisher = rospy.Publisher("/jebediah/Action", Action, latch=True, queue_size=1)
        rospy.Subscriber("/jebediah/State", State, self.state_callback)
        rospy.loginfo("Done..")
        self.control = False
    
    def reset(self):
        self.resetService = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        try:
            e = self.resetService()
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s", e)
        rospy.loginfo("Simulation reset")

    def eval_period(self, s, T, time):
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
            # print T, val, interpolate(val, np.array([s,period]))
            return interpolate(val, np.array([s,period]))
    
    @timeit
    def state_callback(self, cb_state):
        # get state
        self.state = cb_state
        if self.control:
            # Calculate error
            # error = np.sqrt((cb_state.Twist.linear.x - self.set_point_linear_velocity[0])**2 + (cb_state.Twist.linear.y - self.set_point_linear_velocity[1])**2 + (cb_state.IMU.angular_velocity.z - self.set_point_angular_velocity)**2)
            # evaluate on setpoint
            action = []
            current_time = time.time()
            # Linear in X
            if self.set_point[0] > self.set_point[2]:
                for signal in self.linear_ref:
                    action.append(self.eval_period(self.linear_ref[signal], self.T_lin, current_time))
            # Angular in Z
            else:
                for signal in self.angular_ref:
                    action.append(self.eval_period(self.angular_ref[signal], self.T_ang, current_time))
                    
            self.set_joints(action)
            
    def get_state(self):
        return self.state

    def set_joints(self, pList, mode='rad'):
        if 'deg' in mode:
            pList = [radians(p) for p in pList]
        if len(pList) != 12:
            rospy.logerr("Joints action must be a vector with 12 positions.")
        else:
            self.action_publisher.publish(pList)

    def set_initial(self):
        self.set_joints([0,0,0,0,0,0,0,0,0,0,0,0])

    def set_helloWorld(self):
        self.set_joints([   0.0,    -45.0,  45.0,
                          -30.0,    -45.0,  45.0,
                            0.0,    -45.0,  45.0,
                           30.0,    -45.0,  45.0], mode='deg')
        time.sleep(0.05)
        for i in range(-45, 80):
            time.sleep(0.005)
            self.set_joints([   0.0, i, 45.0,
                              -30.0,-45.0,45.0,
                                0.0,-45.0,45.0,
                               30.0,-45.0,45.0], mode='deg')
        time.sleep(0.5)
        for i in range(0, 30):
            time.sleep(0.005)
            self.set_joints([   i, 80.0,45.0,
                            -30.0,-45.0,45.0,
                              0.0,-45.0,45.0,
                             30.0,-45.0,45.0], mode='deg')
        # wave
        for k in range(0, 5):
            for i in range(30, -30):
                time.sleep(0.005)
                self.set_joints([   i, 80.0,45.0,
                                -30.0,-45.0,45.0,
                                  0.0,-45.0,45.0,
                                 30.0,-45.0,45.0], mode='deg')
            for i in range(-30, 30):
                time.sleep(0.005)
                self.set_joints([   i, 80.0,45.0,
                                -30.0,-45.0,45.0,
                                  0.0,-45.0,45.0,
                                 30.0,-45.0,45.0], mode='deg')

    def set_control_loop(self):
        # Load reference signals
        path = os.path.dirname(os.path.realpath(__file__))
        self.linear_ref = pd.read_csv(path + '/model/reference/forward.csv')
        self.angular_ref = pd.read_csv(path + '/model/reference/rotate.csv')
        self.coefs = [-2.15245784, 4.86364516, -10.37872853, 5.01919578]
        self.T_lin = np.exp(self.coefs[1]) * np.exp(self.coefs[0]*self.set_point[0])
        self.T_ang = np.exp(self.coefs[3]) * np.exp(self.coefs[2]*self.set_point[2])
        self.control = True

        
def main(args):
    try:
        j = jcontroller()
        j.set_initial()
        time.sleep(1)
        j.reset()
        j.set_control_loop()
        time.sleep(100)
        #j.set_helloWorld()
        # j.set_joints([40.0, 40.0, 40.0,
        #                 40.0, 40.0, 40.0,
        #                 40.0, 40.0, 40.0,
        #                 40.0, 40.0, 40.0], mode='deg')
        # while not rospy.is_shutdown():
        #     time.sleep(1)
        #     rospy.loginfo(j.get_state())
        #     print j.get_state()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
   main(None)