#!/usr/bin/env python

import rospy
import time
import math
import re
import numpy as np
from collections import OrderedDict

from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.msg import ContactState, ContactsState, ModelStates
from control_msgs.msg import JointControllerState
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
    # Sensors
    ground = [False, False, False, False]
    selfCollide = [False, False, False, False]
    angles = [  [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]
    state = ''
    # Joints sensor noise
    mu = 0
    sigma = 0.005
    def __init__(self):
        rospy.init_node('jebediah_controler')
        # Create reset service
        rospy.loginfo("Waiting for gazebo services")
        rospy.wait_for_service('gazebo/reset_simulation')
        # Create publisher to all topics
        for c in self.servos:
            topicName = "jebediah/" + c + "/command"
            self.servos[c] = rospy.Publisher(topicName,Float64, latch=True, queue_size=1)
        rospy.loginfo("Created publishers")
        # Subscribe to sensor topics
        rospy.Subscriber("jebediah/tibia_1_conctact_sensor", ContactsState, self.callback_tibia_sensor)
        rospy.Subscriber("jebediah/tibia_2_conctact_sensor", ContactsState, self.callback_tibia_sensor)
        rospy.Subscriber("jebediah/tibia_3_conctact_sensor", ContactsState, self.callback_tibia_sensor)
        rospy.Subscriber("jebediah/tibia_4_conctact_sensor", ContactsState, self.callback_tibia_sensor)

        # Model state
        rospy.Subscriber("gazebo/model_states", ModelStates, self.callback_body_state)
        # Link angles
        rospy.Subscriber("/jebediah/coxa_1_position_controller/state", JointControllerState, self.callback_joint_state_00)
        rospy.Subscriber("/jebediah/coxa_2_position_controller/state", JointControllerState, self.callback_joint_state_10)
        rospy.Subscriber("/jebediah/coxa_3_position_controller/state", JointControllerState, self.callback_joint_state_20)
        rospy.Subscriber("/jebediah/coxa_4_position_controller/state", JointControllerState, self.callback_joint_state_30)
        rospy.Subscriber("/jebediah/femur_1_position_controller/state", JointControllerState, self.callback_joint_state_01)
        rospy.Subscriber("/jebediah/femur_2_position_controller/state", JointControllerState, self.callback_joint_state_11)
        rospy.Subscriber("/jebediah/femur_3_position_controller/state", JointControllerState, self.callback_joint_state_21)
        rospy.Subscriber("/jebediah/femur_4_position_controller/state", JointControllerState, self.callback_joint_state_31)
        rospy.Subscriber("/jebediah/tibia_1_position_controller/state", JointControllerState, self.callback_joint_state_02)
        rospy.Subscriber("/jebediah/tibia_2_position_controller/state", JointControllerState, self.callback_joint_state_12)
        rospy.Subscriber("/jebediah/tibia_3_position_controller/state", JointControllerState, self.callback_joint_state_22)
        rospy.Subscriber("/jebediah/tibia_4_position_controller/state", JointControllerState, self.callback_joint_state_32)
        
        rospy.loginfo("Controller API loaded")

#   def callback_joint_state_lj(self, joint):
    # LEG 1
    def callback_joint_state_00(self, joint):
        self.angles[0][0] = joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_01(self, joint):
        self.angles[0][1] = -1*joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_02(self, joint):
        self.angles[0][2] = -1*joint.process_value + np.random.normal(self.mu, self.sigma)
    # LEG 2
    def callback_joint_state_10(self, joint):
        self.angles[1][0] = joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_11(self, joint):
        self.angles[1][1] = joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_12(self, joint):
        self.angles[1][2] = joint.process_value + np.random.normal(self.mu, self.sigma)
    # LEG 3
    def callback_joint_state_20(self, joint):
        self.angles[2][0] = joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_21(self, joint):
        self.angles[2][1] = joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_22(self, joint):
        self.angles[2][2] = joint.process_value + np.random.normal(self.mu, self.sigma)
    # LEG 4
    def callback_joint_state_30(self, joint):
        self.angles[3][0] = joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_31(self, joint):
        self.angles[3][1] = -1*joint.process_value + np.random.normal(self.mu, self.sigma)
    def callback_joint_state_32(self, joint):
        self.angles[3][2] = -1*joint.process_value + np.random.normal(self.mu, self.sigma)
#       self.angles[l][j] = joint.process_value + np.random.normal(self.mu, self.sigma)

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

    def get_joints(self):
        degree = []
        for leg in self.angles:
            degree.append([math.degrees(j) for j in leg])
        return degree

    def set_joints(self, pList):
        
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
        self.set_joints([[   0.0,    -45.0,  45.0],
                        [ -30.0,    -45.0,  45.0],
                        [   0.0,    -45.0,  45.0],
                        [  30.0,    -45.0,  45.0]])
        time.sleep(0.1)
        for i in range(-45, 80):
            time.sleep(0.001)
            self.set_joints([[0.0, i, 45.0],
                        [-30.0,-45.0,45.0],
                        [0.0,-45.0,45.0],
                        [30.0,-45.0,45.0]])
        time.sleep(0.5)
        for i in range(0, 30):
            time.sleep(0.001)
            self.set_joints([[i, 80.0,45.0],
                        [-30.0,-45.0,45.0],
                        [0.0,-45.0,45.0],
                        [30.0,-45.0,45.0]])
        # wave
        for k in range(0, 5):
            for i in range(30, -30):
                time.sleep(0.002)
                self.set_joints([[i, 80.0,45.0],
                        [-30.0,-45.0,45.0],
                        [0.0,-45.0,45.0],
                        [30.0,-45.0,45.0]])
            for i in range(-30, 30):
                time.sleep(0.002)
                self.set_joints([[i, 80.0,45.0],
                        [-30.0,-45.0,45.0],
                        [0.0,-45.0,45.0],
                        [30.0,-45.0,45.0]])

if __name__ == '__main__':
    try:
        j = jcontroller()
        j.set_initial()
        j.reset()
        j.set_helloWorld()
        j.set_joints([[40.0, 40.0, 40.0],
                        [40.0, 40.0, 40.0],
                        [40.0, 40.0, 40.0],
                        [40.0, 40.0, 40.0]])
        while True:
            for leg in j.get_joints():
                print leg
            print '='*50
            time.sleep(1)
    except rospy.ROSInterruptException:
        pass
