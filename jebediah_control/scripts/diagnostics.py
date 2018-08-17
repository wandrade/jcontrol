#! /usr/bin/env python
import rospy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import signal
import sys

from jcontrol_msgs.msg import State, Action

buffer_size = 1500
topic_p = 0.02
refresh_rate = 0.5

stop = False
def signal_handler(signal, frame):
    print 'You pressed Ctrl+C - or killed me with -2'
    stop = True
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

states = []
def state_callback(data):
    states.append(data)
    if len(states) >= buffer_size:
        del states[0]

actions = [[0]*12]*buffer_size
def action_callback(data):
    actions.append(list(data.Actions))
    del actions[0]

def main(args):
    rospy.init_node('diagnostics')
    # subscribe Action topic
    rospy.Subscriber("/jebediah/State", State, state_callback)
    rospy.Subscriber("/jebediah/Action", Action, action_callback)
    # Plot loop
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # time vector
    t = np.linspace(start=0, stop=buffer_size*topic_p, num=len(actions))
    
    while(1):
        # draw
        if stop:
            break
        else:
            fig.canvas.draw()
            # time.sleep(refresh_rate)
            plt.gcf().clear()
            orientation = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
            velocity = plt.subplot2grid((4,4),(0,3))
            leg1 = plt.subplot2grid((4,4), (3,0))
            leg2 = plt.subplot2grid((4,4), (3,1))
            leg3 = plt.subplot2grid((4,4), (3,2))
            leg4 = plt.subplot2grid((4,4), (3,3))
        
        # Data vector
        actions_plot = zip(*actions)
        # plot
        velocity.scatter([0],[0])
        velocity.arrow(0, 0, states[-1].Twist.linear.x, states[-1].Twist.linear.y)
        velocity.set_xlim([-1,1])
        velocity.set_ylim([-1,1])
        velocity.grid()

        leg1.plot(t, actions_plot[0], 'r')
        leg1.plot(t, actions_plot[1], 'g')
        leg1.plot(t, actions_plot[2], 'b')
        leg1.set_ylim([-3.5,3.5])
        
        leg2.plot(t, actions_plot[3], 'r')
        leg2.plot(t, actions_plot[4], 'g')
        leg2.plot(t, actions_plot[5], 'b')
        leg2.set_ylim([-3.5,3.5])
        
        leg3.plot(t, actions_plot[6], 'r')
        leg3.plot(t, actions_plot[7], 'g')
        leg3.plot(t, actions_plot[8], 'b')
        leg3.set_ylim([-3.5,3.5])
        
        leg4.plot(t, actions_plot[9], 'r')
        leg4.plot(t, actions_plot[10], 'g')
        leg4.plot(t, actions_plot[11], 'b')
        leg4.set_ylim([-3.5,3.5])


if __name__ == '__main__':
   main()