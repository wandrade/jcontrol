#!/usr/bin/env python

import rospy
#from osrf_msgs.msg import JointCommands
from sensor_msgs.msg import JointState

currentJointState = JointState()
def jointStatesCallback(msg):
  global currentJointState
  currentJointState = msg

def main():
    rospy.init_node('controler')
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass