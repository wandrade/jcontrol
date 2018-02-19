#!/usr/bin/env python

import rospy
import time
import math
import re

from collections import OrderedDict

from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.msg import ContactState, ContactsState

class jcontroller:
    # Controllers names
    servos = {"coxa_1_position_controller":None,
            "femur_1_position_controller":None,
            "tibia_1_position_controller":None,
            "coxa_2_position_controller":None,
            "femur_2_position_controller":None,
            "tibia_2_position_controller":None,
            "coxa_3_position_controller":None,
            "femur_3_position_controller":None,
            "tibia_3_position_controller":None,
            "coxa_4_position_controller":None,
            "femur_4_position_controller":None,
            "tibia_4_position_controller":None}
    ground = [False, False, False, False]
    selfCollide = [False, False, False, False]
    
    def __init__(self):
        rospy.init_node('jebediah_controler')
        # Create reset service
        rospy.loginfo("Waiting for gazebo services")
        rospy.wait_for_service('gazebo/reset_simulation')
        # Create publisher to all topics
        for c in self.servos:
            topicName = "jebediah/"+c+"/command"
            self.servos[c] = rospy.Publisher(topicName,Float64, latch=True, queue_size=1)
        rospy.loginfo("Created publishers")
        # Subscribe to sensor topics
        rospy.Subscriber("jebediah/tibia_1_conctact_sensor", ContactsState, self.callback_tibia_sensor)
        rospy.Subscriber("jebediah/tibia_2_conctact_sensor", ContactsState, self.callback_tibia_sensor)
        rospy.Subscriber("jebediah/tibia_3_conctact_sensor", ContactsState, self.callback_tibia_sensor)
        rospy.Subscriber("jebediah/tibia_4_conctact_sensor", ContactsState, self.callback_tibia_sensor)
        rospy.loginfo("Controller API loaded")

    def callback_tibia_sensor(self, data):
        # identify wich tibia called
        tibia = data.header.frame_id
        # get tibia number
        tibia = int(re.search(r'\d+', tibia).group())-1
        # count collision types
        flag_ground = 0
        flag_self = 0
        for s in data.states:
            if 'ground' in s.collision2_name: # This means tibia on ground verticaly
                if s.total_wrench.force.z > s.total_wrench.force.x and s.total_wrench.force.z > s.total_wrench.force.y:
                    flag_ground = flag_ground + 1
            else: # This means self colission
                    flag_self = flag_self + 1
        self.ground[tibia] = bool(flag_ground)
        self.selfCollide[tibia] = bool(flag_self)

    def reset(self):
        self.resetService = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        try:
            e = self.resetService()
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s", e)
        rospy.loginfo("Simulation reset")
    
    def get_selfcollide(self):
        return any(x for x in self.selfCollide if x)
    
    def get_ground(self):
        return self.ground

    def setJoints(self, pList):
        if len(pList) < len(self.servos):
            rospy.logerr("Position list must be of same length as servos number.")
            return -1
        #fix references angle problems
        pList[4] = -pList[4]
        pList[7] = -pList[7]
        pList[8] = -pList[8]
        pList[11] = -pList[11]
        pList = [math.radians(p) for p in pList]
        for i, controller in enumerate(sorted(self.servos)):
            self.servos[controller].publish(pList[i])

    def set_initial(self):
        # set to initial position
        # Coxa
        for controller in [s for s in self.servos if 'coxa' in s]:
            self.servos[controller].publish(0.0)
        rospy.loginfo("Set coxa's servos to 0")
        time.sleep(0.5)
        # Tibia
        for controller in [s for s in self.servos if 'tibia' in s]:
            self.servos[controller].publish(0.0)
        rospy.loginfo("Set tibia's servos to 0")
        time.sleep(0.5)
        # Femur
        for controller in [s for s in self.servos if 'femur' in s]:
            self.servos[controller].publish(0.0)
        rospy.loginfo("Set femur's servos to 0")
        time.sleep(3)

    def set_helloWorld(self):
        self.setJoints([0.0, -30.0, 0.0, 30.0, 
                -45.0, -45.0, -45.0, -45.0,
                45.0, 45.0, 45.0, 45.0])
        for i in range(-45, 80):
            time.sleep(0.001)
            self.setJoints([0.0, -30.0, 0.0, 30.0, 
                        i, -45.0, -45.0, -45.0,
                        45.0, 45.0, 45.0, 45.0])
        time.sleep(0.5)
        for i in range(0, 30):
            time.sleep(0.001)
            self.setJoints([i, -30.0, 0.0, 30.0, 
                        80, -45.0, -45.0, -45.0,
                        45.0, 45.0, 45.0, 45.0])
        for k in range(0, 5):
            for i in range(30, -30):
                time.sleep(0.002)
                self.setJoints([i, -30.0, 0.0, 30.0, 
                            80, -45.0, -45.0, -45.0,
                            45.0, 45.0, 45.0, 45.0])
            for i in range(-30, 30):
                time.sleep(0.002)
                self.setJoints([i, -30.0, 0.0, 30.0, 
                            80, -45.0, -45.0, -45.0,
                            45.0, 45.0, 45.0, 45.0])

if __name__ == '__main__':
    try:
        j = jcontroller()
        j.reset()
        j.set_initial()
        j.set_helloWorld()
    except rospy.ROSInterruptException:
        pass