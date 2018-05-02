#!/usr/bin/env python

from controller import jcontroller
import pymatlab
import time
import numpy as np
#def main():
    # initiate robot
j = jcontroller()
    # Start matlab session
    
    # gait angles list
matSess=pymatlab.session_factory()
matSess.run("addpath('~/Omnidirectional-Static-Walking-of-Quadruped/')")
def joinLists(a, b, steps = 30):
    # generate intermediary stepts
    c = []
    loa = [] # list of actuators
    ini = a[-1]
    end = b[0]
    for i in range (4): # for each leg
        for j in range(3): # for each motor
            step =  (end[i][j]-ini[i][j])/steps
            if step == 0:
                loa.append([ini[i][j]] * steps)
            else:
                loa.append(np.arange(ini[i][j], end[i][j], step).tolist())
    for i in range(steps):
        c.append([
            [loa[0][i], loa[1][i], loa[2][i]],
            [loa[3][i], loa[4][i], loa[5][i]],
            [loa[6][i], loa[7][i], loa[8][i]],
            [loa[9][i], loa[10][i], loa[11][i]]
        ])
    return a + c + b

def genLongPath():
    cmd = str("[dist, ang] = GenerateGait(4, 1, 0.1, 0.09, 0, 0, 100)")
    matSess.run(cmd)
    angList1 = matSess.getvalue('ang').tolist()
    #
    cmd = str("[dist, ang] = GenerateAngularGait(0, 1, 0.1, 0.09, 0, 0, 100)")
    matSess.run(cmd)
    angList2 = matSess.getvalue('ang').tolist()
    #
    cmd = str("[dist, ang] = GenerateGait(1, 1, 0.2, 0.07, 0, 0, 100)")
    matSess.run(cmd)
    angList3 = matSess.getvalue('ang').tolist()
    #
    cmd = str("[dist, ang] = GenerateGait(3, 1, 0.2, 0.2, 0, 0, 200)")
    matSess.run(cmd)
    angList4 = matSess.getvalue('ang').tolist()
    #
    cmd = str("[dist, ang] = GenerateAngularGait(1, 1, 0.15, 0.16, 0, 0, 100)")
    matSess.run(cmd)
    angList5 = matSess.getvalue('ang').tolist()
    #
    cmd = str("[dist, ang] = GenerateGait(4, 1, 0.3, 0.09, 0, 0, 80)")
    matSess.run(cmd)
    angList6 = matSess.getvalue('ang').tolist()
    #
    cmd = str("[dist, ang] = GenerateAngularGait(1, 1, 0.15, 0.09, 0, 0, 100)")
    matSess.run(cmd)
    angList7 = matSess.getvalue('ang').tolist()
    #
    cmd = str("[dist, ang] = GenerateGait(0, 1, 0.2, 0.09, 0, 0, 100)")
    matSess.run(cmd)
    angList8 = matSess.getvalue('ang').tolist()
    #
    temp1 = joinLists(angList1, angList2)
    temp2 = joinLists(angList3, angList4)
    temp3 = joinLists(angList5, angList6)
    temp4 = joinLists(angList7, angList8)
    tmp1 = joinLists(temp1, temp2)
    tmp2 = joinLists(temp3, temp4)
    angList = joinLists(tmp1, tmp2)
    #
    return angList

angList = genLongPath()


def run(angList):
    # (direction, gait, walkDistance, bh, turn, plot)
    j.setJoints(angList[0])
    j.reset()
    for step in angList:
        j.setJoints(step)
        time.sleep(0.02)

print('Starting in 2s...')
time.sleep(4)
run(angList)



# if __name__ == '__main__':
#     main()