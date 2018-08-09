#!/usr/bin/env python

import rospy
from jcontrol_msgs.msg import State, Action
from std_srvs.srv import Empty
import neuralnetwork
import time
from math import *
import numpy as np

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
    set_point_linear_velocity = [0.08, 1.08]
    set_point_angular_velocity = 1.08
    action_publisher = None
    state = None
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
    
    # @timeit
    def state_callback(self, cb_state):
        # get state
        self.state = cb_state
        if self.control:
            # Calculate error
            error = [
                cb_state.Twist.linear.x - self.set_point_linear_velocity[0],
                cb_state.Twist.linear.y - self.set_point_linear_velocity[1],
                cb_state.IMU.angular_velocity.z - self.set_point_angular_velocity
            ] 
            # Create eval vector
            eval_vector = np.array([[
                self.set_point_linear_velocity[0],
                self.set_point_linear_velocity[1],
                self.set_point_angular_velocity,
                amap(cb_state.Angles[0], -90, 90, -1, 1),
                amap(cb_state.Angles[1], -90, 90, -1, 1),
                amap(cb_state.Angles[2], -90, 90, -1, 1),
                amap(cb_state.Angles[3], -90, 90, -1, 1),
                amap(cb_state.Angles[4], -90, 90, -1, 1),
                amap(cb_state.Angles[5], -90, 90, -1, 1),
                amap(cb_state.Angles[6], -90, 90, -1, 1),
                amap(cb_state.Angles[7], -90, 90, -1, 1),
                amap(cb_state.Angles[8], -90, 90, -1, 1),
                amap(cb_state.Angles[9], -90, 90, -1, 1),
                amap(cb_state.Angles[10], -90, 90, -1, 1),
                amap(cb_state.Angles[11], -90, 90, -1, 1),
                cb_state.Ground_Collision[0],
                cb_state.Ground_Collision[1],
                cb_state.Ground_Collision[2],
                cb_state.Ground_Collision[3],
                cb_state.IMU.orientation.x,
                cb_state.IMU.orientation.y,
                cb_state.IMU.orientation.z,
                cb_state.IMU.orientation.w,
                cb_state.IMU.angular_velocity.x,
                cb_state.IMU.angular_velocity.y,
                cb_state.IMU.angular_velocity.z,
                cb_state.IMU.linear_acceleration.x,
                cb_state.IMU.linear_acceleration.y,
                cb_state.IMU.linear_acceleration.z,
                cb_state.Twist.linear.x,
                cb_state.Twist.linear.y,
                cb_state.Twist.linear.z
            ]])
            # Eval next step
            action = self.nn.predict(eval_vector)[0].tolist()
            # map from -1 1 to radians
            action = [amap(alpha, -1, 1, -1.57079, 1.57079) for alpha in action]
            # Publish action
            self.set_joints(action)
            # verbose
            # print error
            # print action
            

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
        # Load Model
        handler = neuralnetwork.neuralNet()
        rospy.loginfo("Loading model from path: %s"%handler.log_path)
        handler.load_model(path=handler.log_path)
        rospy.loginfo("Model loaded...")
        handler.model
        self.nn = handler
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
   main()