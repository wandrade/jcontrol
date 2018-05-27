#!/usr/bin/env python

import rospy
from jcontrol_msgs.msg import State, Action
from std_srvs.srv import Empty
import time
class jcontroller:
    action_publisher = None
    state = None
    def __init__(self):
        rospy.init_node('jebediah_controler')
        # Create reset service
        rospy.loginfo("Waiting for gazebo services")
        rospy.wait_for_service('gazebo/reset_simulation')
        self.action_publisher = rospy.Publisher("/jebediah/Action", Action, latch=True, queue_size=1)
        rospy.Subscriber("/jebediah/State", State, self.state_callback)
        
    def reset(self):
        self.resetService = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        try:
            e = self.resetService()
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s", e)
        rospy.loginfo("Simulation reset")
    
    def state_callback(self, cb_state):
        # get state
        self.state = cb_state
        # calculate controll
        # publish actions

    def get_state(self):
        return self.state

    def set_joints(self, pList):
        if len(pList) < 12:
            rospy.logerr("Joints action must be a vector with 12 positions.")
        else:
            self.action_publisher.publish(pList)


    def set_initial(self):
        self.set_joints([0,0,0,0,0,0,0,0,0,0,0,0])

    def set_helloWorld(self):
        self.set_joints([   0.0,    -45.0,  45.0,
                          -30.0,    -45.0,  45.0,
                            0.0,    -45.0,  45.0,
                           30.0,    -45.0,  45.0])
        time.sleep(0.1)
        for i in range(-45, 80):
            time.sleep(0.001)
            self.set_joints([   0.0, i, 45.0,
                              -30.0,-45.0,45.0,
                                0.0,-45.0,45.0,
                               30.0,-45.0,45.0])
        time.sleep(0.5)
        for i in range(0, 30):
            time.sleep(0.001)
            self.set_joints([   i, 80.0,45.0,
                            -30.0,-45.0,45.0,
                              0.0,-45.0,45.0,
                             30.0,-45.0,45.0])
        # wave
        for k in range(0, 5):
            for i in range(30, -30):
                time.sleep(0.002)
                self.set_joints([   i, 80.0,45.0,
                                -30.0,-45.0,45.0,
                                  0.0,-45.0,45.0,
                                 30.0,-45.0,45.0])
            for i in range(-30, 30):
                time.sleep(0.002)
                self.set_joints([   i, 80.0,45.0,
                                -30.0,-45.0,45.0,
                                  0.0,-45.0,45.0,
                                 30.0,-45.0,45.0])

def main(args):
    try:
        j = jcontroller()
        j.set_initial()
        j.reset()
        j.set_helloWorld()
        j.set_joints([-90.0, 40.0, 40.0,
                        40.0, 40.0, 40.0,
                        40.0, 40.0, 40.0,
                        40.0, 40.0, 40.0])
        while not rospy.is_shutdown():
            time.sleep(1)
            rospy.loginfo(j.get_state())
            print j.get_state()
    except rospy.ROSInterruptException:
        pass
if __name__ == '__main__':
   main()
