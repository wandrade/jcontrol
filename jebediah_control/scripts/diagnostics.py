#! /usr/bin/env python
import rospy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import signal
import sys
from jcontrol_msgs.msg import State, Action, SetPoint

buffer_size = 1500
topic_p = 0.02
refresh_rate = 0.25
set_point = [0.0, 0.0, 0.0]
def setpoint_callback( sp):
    set_point[0] = sp.Linear[0]
    set_point[1] = sp.Linear[1]
    set_point[2] = sp.Angular

stop = False
def signal_handler(signal, frame):
    print 'You pressed Ctrl+C - or killed me with -2'
    stop = True
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

states = []
y_vel = [0]*buffer_size
x_vel = [0]*buffer_size
a_vel = [0]*buffer_size
def state_callback(data):
    states.append(data)
    x_vel.append(data.Twist.linear.x)
    del(x_vel[0])
    y_vel.append(data.Twist.linear.y)
    del(y_vel[0])
    a_vel.append(data.IMU.angular_velocity.z)
    del(a_vel[0])

    if len(states) >= buffer_size:
        del states[0]

actions = [[0]*12]*buffer_size
def action_callback(data):
    actions.append(list(data.Actions))
    del actions[0]
def ewma(data, window = 50):
    if type(data) is not np.ndarray:
        data = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def main(args):
    rospy.init_node('diagnostics')
    # subscribe Action topic
    rospy.Subscriber("/jebediah/State", State, state_callback)
    rospy.Subscriber("/jebediah/Action", Action, action_callback)
    rospy.Subscriber("/jebediah/SetPoint", SetPoint, setpoint_callback)
    # Plot loop
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # time vector
    t = np.linspace(start=0, stop=buffer_size*topic_p, num=len(actions))
    
    orientation = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
    l_velocity = plt.subplot2grid((4,4),(0,3))
    velocity = plt.subplot2grid((4,4),(1,3))
    a_velocity = plt.subplot2grid((4,4),(2,3))
    leg1 = plt.subplot2grid((4,4), (3,0))
    leg2 = plt.subplot2grid((4,4), (3,1))
    leg3 = plt.subplot2grid((4,4), (3,2))
    leg4 = plt.subplot2grid((4,4), (3,3))

    while(not stop):
        plt.tight_layout()
        plt.draw()
        plt.pause(refresh_rate)
        
        # Data vector
        actions_plot = zip(*actions)
        
        module = np.sqrt(set_point[0]**2 + set_point[1]**2)
        if not module: module = 0.03
        l_velocity.cla()
        l_velocity.arrow(0, 0, np.mean(x_vel[-50:]), np.mean(y_vel[-50:]), color='r', head_width=module/5)
        l_velocity.arrow(0, 0, set_point[0], set_point[1], color='k', head_width=module/5, linestyle='dotted', alpha=0.7)
        l_velocity.set_xlim([-2*module,2*module])
        l_velocity.set_ylim([-2*module,2*module])
        l_velocity.set_aspect('equal')
        l_velocity.grid()
        l_velocity.set_title("Velocity vector")

        # vel module
        # module = sqrt(np.array(states[-50:-1].Twist.linear.x)**2 + np.array(states[-50:-1].Twist.linear.y)**2)
        velocity.cla()
        velocity.plot(t, x_vel, 'g', alpha=0.15)
        velocity.plot(t, ewma(x_vel), 'g', label="X", alpha=0.7)
        velocity.plot(t, y_vel, 'b', alpha=0.15)
        velocity.plot(t, ewma(y_vel), 'b', label="Y", alpha=0.7)
        velocity.axhline(set_point[0], color='g', linestyle='dotted', label="Y sp")
        velocity.axhline(set_point[1], color='b', linestyle='dotted', label="z sp")
        velocity.legend(loc=3)
        velocity.grid()
        velocity.set_title("Linear velocity")

        a_velocity.cla()
        a_velocity.plot(t, a_vel, 'm', alpha=0.15)
        a_velocity.plot(t, ewma(a_vel), 'm', label="Ang Vel", alpha=0.7)
        a_velocity.axhline(set_point[2], color='m', linestyle='dotted', label="Y sp")
        a_velocity.grid()
        a_velocity.set_title("Angular velocity")


        leg1.cla()
        leg1.plot(t, actions_plot[0], 'r', label="Coxa")
        leg1.plot(t, actions_plot[1], 'g', label="Femur")
        leg1.plot(t, actions_plot[2], 'b', label="Tibia")
        leg1.set_ylim([-3.5, 3.5])
        leg1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2 ,np.pi])
        leg1.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
        leg1.grid()
        leg1.set_ylabel("Angular position")
        leg1.set_title("Leg 1")
        leg1.legend(loc=3)
        
        leg2.cla()
        leg2.plot(t, actions_plot[3], 'r')
        leg2.plot(t, actions_plot[4], 'g')
        leg2.plot(t, actions_plot[5], 'b')
        leg2.set_ylim([-3.5,3.5])
        leg2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2 ,np.pi])
        leg2.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
        leg2.grid()
        leg2.set_title("Leg 2")
        
        leg3.cla()
        leg3.plot(t, actions_plot[6], 'r')
        leg3.plot(t, actions_plot[7], 'g')
        leg3.plot(t, actions_plot[8], 'b')
        leg3.set_ylim([-3.5,3.5])
        leg3.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2 ,np.pi])
        leg3.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
        leg3.grid()
        leg3.set_title("Leg 3")
        
        leg4.cla()
        leg4.plot(t, actions_plot[9], 'r')
        leg4.plot(t, actions_plot[10], 'g')
        leg4.plot(t, actions_plot[11], 'b')
        leg4.set_ylim([-3.5,3.5])
        leg4.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2 ,np.pi])
        leg4.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
        leg4.grid()
        leg4.set_title("Leg 4")


if __name__ == '__main__':
   main(None)