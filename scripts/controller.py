#!/usr/bin/env python

import rospy
import time
import math
import re

from collections import OrderedDict

from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.msg import ContactState, ContactsState, ModelStates

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
    state = ''
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
        # State topic
        rospy.Subscriber("gazebo/model_states", ModelStates, self.callback_body_state)
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

    def callback_body_state(self, states):
        self.state = states.pose[1]

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

    def get_state(self):
        return self.state

    def setJoints(self, pList):
        
        if len(pList) < 4:
            rospy.logerr("Position list must be of same length as servos number.")
            return -1
        #fix references angle problems
        pList[0][1] = -pList[0][1]
        pList[0][2] = -pList[0][2]
        pList[3][1] = -pList[3][1]
        pList[3][2] = -pList[3][2]
        for i, leg in enumerate(pList):
            leg = [math.radians(p) for p in leg]
            
            self.servos[str('coxa_%d_position_controller' % (i+1))].publish(leg[0])
            self.servos[str('femur_%d_position_controller' % (i+1))].publish(leg[1])
            self.servos[str('tibia_%d_position_controller' % (i+1))].publish(leg[2])

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

    def set_helloWorld(self):
        self.setJoints([[90.0, 45.0,135.0],
                        [60.0,45.0,135.0],
                        [90.0,45.0,135.0],
                        [120.0,45.0,135.0]])
        time.sleep(0.1)
        for i in range(135, 170):
            time.sleep(0.001)
            self.setJoints([[90.0, i,135.0],
                        [60.0,45.0,135.0],
                        [90.0,45.0,135.0],
                        [120.0,45.0,135.0]])
        time.sleep(0.5)
        for i in range(90, 120):
            time.sleep(0.001)
            self.setJoints([[i, 170.0,135.0],
                        [60.0,45.0,135.0],
                        [90.0,45.0,135.0],
                        [120.0,45.0,135.0]])
        # wave
        for k in range(0, 5):
            for i in range(120, 60):
                time.sleep(0.002)
                self.setJoints([[i, 170.0,135.0],
                        [60.0,45.0,135.0],
                        [90.0,45.0,135.0],
                        [120.0,45.0,135.0]])
            for i in range(60, 120):
                time.sleep(0.002)
                self.setJoints([[i, 170.0,135.0],
                        [60.0,45.0,135.0],
                        [90.0,45.0,135.0],
                        [120.0,45.0,135.0]])

if __name__ == '__main__':
    try:
        j = jcontroller()
        j.set_initial()
        j.reset()
        j.setJoints([[40.0, 40.0, 40.0],
                        [40.0, 40.0, 40.0],
                        [40.0, 40.0, 40.0],
                        [40.0, 40.0, 40.0]])
        j.set_helloWorld()
    except rospy.ROSInterruptException:
        pass