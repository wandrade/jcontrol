#!/usr/bin/env python
from __future__ import division

import rospy
from jcontrol_msgs.msg import State, Action, SetPoint
from std_srvs.srv import Empty
# import neuralnetwork
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
    set_point = [0.0, 0.0, 0.0]
    action_publisher = None
    state = None
    prev_actions = [2]*12
    coefs = [-2.15245784, 4.86364516, -10.37872853, 5.01919578]
    
    def __init__(self, simulation=False):
        rospy.init_node('jebediah_controler')
        # Create reset service
        rospy.Subscriber("/jebediah/State", State, self.state_callback)
        rospy.Subscriber("/jebediah/SetPoint", SetPoint, self.setpoint_callback)
        self.action_publisher = rospy.Publisher("/jebediah/Action", Action, latch=True, queue_size=1)
        rospy.loginfo("Done..")
        self.control = False
        self.sim = simulation
        if simulation:
            rospy.loginfo("Waiting for gazebo services")
            rospy.wait_for_service('gazebo/reset_simulation')
        else:
            import Adafruit_PCA9685
            self.pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)# Configure min and max servo pulse lengths
            self.servo_min = 90  # Min pulse length out of 4096
            self.servo_max = 490  # Max pulse length out of 4096
            self.frequencie = 50
            # Set frequency to 60hz, good for servos.
            self.pwm.set_pwm_freq(self.frequencie)
            # release motors
            for i in range(16):
                self.pwm.set_pwm(i, 0, 0)

    def set_servo_pulse(self, channel, pulse):
        pulse_length = 1000000    # 1,000,000 us per second
        pulse_length //= self.frequencie       # 60 Hz
        pulse_length //= 4096     # 12 bits of resolution
        pulse *= 1000
        pulse //= pulse_length
        self.pwm.set_pwm(channel, 0, pulse)
    
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
    
    # @timeit
    def setpoint_callback(self, sp):
        print "sp changed"
        self.set_point[0] = sp.Linear[0]
        self.set_point[1] = sp.Linear[1]
        self.set_point[2] = sp.Angular
        self.T_lin = np.exp(self.coefs[1]) * np.exp(self.coefs[0]*self.set_point[0])
        self.T_ang = np.exp(self.coefs[3]) * np.exp(self.coefs[2]*self.set_point[2])

    # @timeit
    def state_callback(self, cb_state):
        tibia_correction = np.pi/2
        # get state
        self.state = cb_state
        if self.control:
            # Calculate error
            # error = np.sqrt((cb_state.Twist.linear.x - self.set_point_linear_velocity[0])**2 + (cb_state.Twist.linear.y - self.set_point_linear_velocity[1])**2 + (cb_state.IMU.angular_velocity.z - self.set_point_angular_velocity)**2)
            # evaluate on setpoint
            action = []
            current_time = time.time()
            # Linear in X
            if self.set_point[0] == 0.0 and self.set_point[1] == 0.0 and self.set_point[2] == 0.0:
                 action = [None]*12
            elif self.set_point[0] > self.set_point[2]:
                for signal in self.linear_ref:
                    action.append(self.eval_period(self.linear_ref[signal], self.T_lin, current_time))
                action[2] = action[2] + tibia_correction
                action[5] = action[5] + tibia_correction
                action[8] = action[8] + tibia_correction
                action[11] = action[11] + tibia_correction
            # Angular in Z
            else:
                for signal in self.angular_ref:
                    action.append(self.eval_period(self.angular_ref[signal], self.T_ang, current_time))
                action[2] = action[2] + tibia_correction
                action[5] = action[5] + tibia_correction
                action[8] = action[8] + tibia_correction
                action[11] = action[11] + tibia_correction
            self.set_joints(action)
            
    def get_state(self):
        return self.state

    def set_joints(self, pList, mode='rad'):
        if len(pList) != 12:
            rospy.logerr("Joints action must be a vector with 12 positions.")
            return None

        if 'deg' in mode:
            pList = [radians(p) if p is not None else None for p in pList]
        
        self.action_publisher.publish(pList)
        if not self.sim: 
            for i in range(12):
                if pList[i] is not None:
                    if i == 1 or i == 4 or i == 7 or i == 10:
                        val = int(amap(pList[i], -np.pi, np.pi, self.servo_min, self.servo_max))
                    else:
                        val = int(amap(pList[i], -np.pi, np.pi, self.servo_max, self.servo_min))
                    self.pwm.set_pwm(i, 0, val)
    
    def set_initial(self):
        # Coxa
        self.set_joints([0, None,None,0,None,None,0,None,None,0,None,None])
        time.sleep(0.5)
        # set femur
        self.set_joints([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    def set_helloWorld(self):
        st = 0.005/3
        # Initial position
        self.set_joints([   0.0,    -80.0,  45.0,
                          -30.0,    -45.0,  45.0,
                            0.0,     15.0,  45.0,
                           30.0,    -45.0,  45.0], mode='deg')
        time.sleep(0.5)
        # Lift first leg femur
        for i in range(-45, 80):
            j = amap(i, -45, 80, 0, 80) #45
            self.set_joints([   None,    i, j,
                                None, None, None,
                                None, None, None,
                                None, None, None], mode='deg')
            time.sleep(st)
        # Position coxa on 0
        for i in range(0, 30):# 30
            self.set_joints([      i, None, None,
                                None, None, None,
                                None, None, None,
                                None, None, None], mode='deg')
            time.sleep(st)
        # wave
        for k in range(3):
            for i in reversed(range(-30, 30)):
                self.set_joints([      i, None, None,
                                    None, None, None,
                                    None, None, None,
                                    None, None, None], mode='deg')
                time.sleep(st)

            for i in range(-30, 30):
                self.set_joints([      i, None, None,
                                    None, None, None,
                                    None, None, None,
                                    None, None, None], mode='deg')
                time.sleep(st)

    def set_control_loop(self):
        # Load reference signals
        path = os.path.dirname(os.path.realpath(__file__))
        self.linear_ref = pd.read_csv(path + '/model/reference/forward.csv')
        self.angular_ref = pd.read_csv(path + '/model/reference/rotate.csv')
        self.T_lin = np.exp(self.coefs[1]) * np.exp(self.coefs[0]*self.set_point[0])
        self.T_ang = np.exp(self.coefs[3]) * np.exp(self.coefs[2]*self.set_point[2])
        self.control = True
        print 'Control loop set'

        
def main(args):
    try:
        j = jcontroller()
        j.set_initial()
        time.sleep(3)
        j.set_helloWorld()
        # j.set_joints([40.0, 40.0, 40.0,
        #                 40.0, 40.0, 40.0,
        #                 40.0, 40.0, 40.0,
        #                 40.0, 40.0, 40.0], mode='deg')
        time.sleep(1)
        j.set_initial()
        # j.reset()
        j.set_control_loop()
        time.sleep(3000)
        #j.set_helloWorld()
        # while not rospy.is_shutdown():
        #     time.sleep(1)
        #     rospy.loginfo(j.get_state())
        #     print j.get_state()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
   main(None)
