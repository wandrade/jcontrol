#!/usr/bin/env python

from controller import jcontroller
import pymatlab
import time

def gaitGen2():
    cmd = """%% Generating Gait Specific Trajectory 

    no_steps = 300; % Number of Steps (Computation)
    ang = zeros(no_steps,4,3);
    % CRAWL:
    % 1 for Y Crawl
    % 2 for X Crawl
    crawl = 2;

    % There will be 2 types of trajectories; Linear and Rotation. 
    % LTrajectory will generate set of points (=no_steps) in space for ...
    % linear motion.
    % Output matrix is 4x3x100 where each matrix contains x, y and z position
    % for all the 4 legs for all 100 interations.

    configs_exp = zeros(4,3);
    tyx_configs = zeros(4,3,4);
    txy_configs = zeros(4,3,4);
    x_configs = zeros(4,3,4);
    y_configs = zeros(4,3,4);
    y_stm = zeros(1,100);
    x_stm = zeros(1,100);
    t_stm = zeros(1,100);

    y_cycles = 1;
    x_cycles = 2;

    y_cont = zeros(4,3,4);  %Continuously move in Y direction for no_cycles times
    y_cont(:,2,:) = 1;

    x_cont = zeros(4,3,4);  %Continuously move in Y direction for no_cycles times
    x_cont(:,1,:) = 1;

    for j = 1:y_cycles
        
    LinearY_Trajectory = LYTrajectory(LookUp(gait,:), Fc, Ls, Hb, no_steps,crawl);



    for i=1:no_steps
        
        ly_footpos = [  0.15 LinearY_Trajectory(1,2,i) LinearY_Trajectory(1,3,i);...
                        0.15 LinearY_Trajectory(2,2,i) LinearY_Trajectory(2,3,i);...
                        0.15 LinearY_Trajectory(3,2,i) LinearY_Trajectory(3,3,i);...
                        0.15 LinearY_Trajectory(4,2,i) LinearY_Trajectory(4,3,i)];
                    
    G2BTmat(2,4) =  ((i*Ls)/(LookUp(gait,5)*no_steps));
    
    
    
    leg_Yangles = inverseKinematics(ly_footpos);

    leg_Yangles2(:,:) = leg_Yangles([4, 2, 1, 3],:,2);
    %leg_Yangles2(:,:) = zeros(4,3)+30;

    leg_Yangles2(:,:) = leg_Yangles2(:,:).*[    1	    1   1
                                                1       1   1
                                                -1	    1   1
                                                -1	    1   1];

    leg_Yangles2(:,:) = leg_Yangles2(:,:)+[     45      0   90
                                                -45     0   90
                                                45      0   90
                                                -45     0   90];

    ang(i,:,:) = leg_Yangles2(:,:);
    y_positions = fwdkinematics(leg_Yangles(:,:,2));
    configs_exp(:,:,i) = ly_footpos;      % Experimental Matrix to check z coordinates
    
    world_Ypos = local2world(y_positions);
    y_configs(:,:,:,i) = world_Ypos + ((j-1)*velocity*CT*y_cont);
    
    end
    end

    """
    return cmd

#def main():
    # initiate robot
j = jcontroller()
    # Start matlab session
    
    # gait angles list
matSess=pymatlab.session_factory()
matSess.run("addpath('~/Omnidirectional-Static-Walking-of-Quadruped/')")
#while True:
def run(direction = 0, gait = 1, distance = 0.5, bodyHeight = 0.09, turn = 0, sleepT= 0.02):
    # (direction, gait, walkDistance, bh, turn, plot)
    cmd = str("[dist, ang] = GenerateGait(%d, %d, %.2f, %.2f, %d, 0)" % (direction, gait, distance, bodyHeight, turn))
    matSess.run(cmd)
    angList = matSess.getvalue('ang').tolist()
    j.setJoints(angList[0])
    j.reset()
    time.sleep(1)
    for step in angList:
        j.setJoints(step)
        time.sleep(sleepT)
        #print step
    print matSess.getvalue('dist')

run(bodyHeight=0.12)
if __name__ == '__main__':
    main()