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
    set_point = [1.5, 0.0, 0.0]
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
            # Calculate second order coeficients for calibration:
            calibration = [[460, 305, 100],
            [125,270,435],
            [420,270,100],
            [440,275,130],
            [92,250,430],
            [465,270,95],
            [440,290,145],
            [110,240,410],
            [470,270,93],
            [430,260,100],
            [85,240,445],
            [420,260,110]]
            self.cal = []
            for c in calibration:
                self.cal.append(np.polyfit([-np.pi/2, 0, np.pi/2], c, 2))
            # print self.cal
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
                
                for i, signal in enumerate(self.linear_ref):
                    action.append(self.eval_period(self.linear_ref[signal], self.T_lin, current_time))
                    if i == 1 or i == 4 or i == 7 or i == 10:
                        action[-1] = action[-1] - 1.5*0.174
                    elif i == 2 or i == 5 or i == 8 or i == 11:
                        action[-1] = action[-1] + 1.5*0.174
                    else:
                        action[-1] = action[-1]*1.1
            # Angular in Z
            else:
                for signal in self.angular_ref:
                    action.append(self.eval_period(self.angular_ref[signal], self.T_ang, current_time))
            # print action
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
                    # if i == 1 or i == 4 or i == 7 or i == 10:
                    #     int(amap(pList[i], -np.pi, np.pi, self.servo_max, self.servo_min))
                    # else:
                    val = int(np.polyval(self.cal[i], pList[i]))
                    self.pwm.set_pwm(i, 0, val)
    
    def set_initial(self):
        # Coxa
        self.set_joints([0, None,None,0,None,None,0,None,None,0,None,None])
        time.sleep(0.2)
        self.set_joints([0, 75, None, 0, 75, 0, None, 75, None, 0, 75, None], mode='deg')
        time.sleep(0.2)
        self.set_joints([0, 75,-45, 0, 75,-45, 0, 75,-45, 0, 75,-45], mode='deg')
        time.sleep(0.5)
        # set femur
        self.set_joints([0, 0, -45, 0, 0, -45, 0, 0, -45, 0, 0, -45], mode='deg')
        time.sleep(0.15)
        self.set_joints([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], mode='deg')
        time.sleep(0.15)
        self.set_joints([0, -20, 0, 0, -20, 0, 0, -20, 0, 0, -20, 0], mode='deg')
        time.sleep(0.8)
        for i in range(5):
            for j in range(4):
                zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # Lift femur 
                zeros[1+3*j] = -3*0.174
                # zeros[2+3*j] = -2*0.174
                self.set_joints(zeros)
                time.sleep(0.05)
                # low femur
                self.set_joints([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                time.sleep(0.05)
            


    def set_helloWorld(self):
        st = 0.0001
        # Initial position
        self.set_joints([   0.0,    -30.0,  20.0,
                          -40.0,      0.0,  0.0,
                            0.0,      0.0,  0.0,
                           40.0,      0.0,  0.0], mode='deg')
        time.sleep(0.2)
        self.set_joints([   0.0,      30.0,  -20,
                          -40.0,      0.0,  0.0,
                            0.0,      0.0,  0.0,
                           40.0,      0.0,  0.0], mode='deg')
        time.sleep(0.2)
        self.set_joints([   0.0,      30.0,  -20,
                          -40.0,      -20.0,  10.0,
                            0.0,      20.0,  0.0,
                           40.0,      -20.0,  10.0], mode='deg')
        time.sleep(0.2)
        # Lift first leg femur
        for i in range(30, 80):
            j = amap(i, 30, 80, -20, 60)
            self.set_joints([   None,    i, j,
                                None, None, None,
                                None, None, None,
                                None, None, None], mode='deg')
            time.sleep(st)
        # Position coxa on 0
        for i in range(0, 30):
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
        time.sleep(0.3)
        self.set_joints([      0,   30,    0,
                            None, None, None,
                            None, None, None,
                            None, None, None], mode='deg')
        time.sleep(0.2)
        self.set_joints([      0,   0,    0,
                               0, None, None,
                               0, None, None,
                               0, None, None], mode='deg')
        time.sleep(0.15)
        self.set_joints([      0, 0, 0,
                               0, 0, 0,
                               0, 0, 0,
                               0, 0, 0], mode='deg')
        time.sleep(0.2)

    def set_control_loop(self):
        # Load reference signals
        path = os.path.dirname(os.path.realpath(__file__))
        self.linear_ref = pd.read_csv(path + '/model/reference/forward.csv')
        self.angular_ref = pd.read_csv(path + '/model/reference/rotate.csv')
        self.T_lin = np.exp(self.coefs[1]) * np.exp(self.coefs[0]*self.set_point[0])
        self.T_ang = np.exp(self.coefs[3]) * np.exp(self.coefs[2]*self.set_point[2])
        # prepare positions
        for i in range (0, 40):
            j = amap(i, 0, 40, 0, -34)
            self.set_joints([   None, i, j,
                                None, i, j,
                                None, i, j,
                                None, i, j,
            ], mode='deg')
            time.sleep(0.01)
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
        time.sleep(3)
        # j.set_initial()
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
