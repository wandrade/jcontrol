%% OMNIDIRECTIONAL STATIC WALKING OF QUADRUPED
% TRAJECTORY PLANNING OF OMNIDIRECTIONAL WALKING OF QUADRUPED BY MINIMIZING
% THE TRANSITION TIME BETWEEN DIFFERENT GAITS

% Summer Internship Project, IIT Jodhpur (2016)
% Author: Nirav Savaliya
% E-mail: niravjwst@gmail.com | GitHub: @Astronirav


%% Defining Basic Variables and Transformations
clc;
clear;
clf;
global BodyDim;  %%Dimensions of Body
global LinkLengths;  %% Link lengths of Leg
global DHParam;  %% DH Parameters of Legs
global G2BTmat;  %% Ground to Body Conter Transformation Matrix
global BC2LTmat; %% Body Center to Leg Transformation Matrix

BodyDim = [0.2895 0.2895]; %% Dimensions: X Y (Width Length)
LinkLengths = [0.047; 0.090; 0.10];
DHParam = [[0;0;0], LinkLengths, [90;0;0]]; % DH Parameters d a alpha

%**************************************************************************
% Body Center to Leg Transformation Matrix
% Body Frame is located at the center of the Body, equidistant...
% from X and Y Dimensions. 
% Transformation is used to move frame to the corners of quadruped...
% where all legs are attached. 
% Frame 1 and 3 have -1 because of the change in direction (-X) relative to
% body frame. Similarly BodyDim(X&Y) are used with appropriate signs. 

BC2LTmat = zeros(4,4,4);
BC2LTmat(:,:,1) = [-1 0 0 -BodyDim(1)/2; 0 1 0 BodyDim(2)/2; 0 0 1 0; 0 0 0 1];
BC2LTmat(:,:,2) = [1 0 0 BodyDim(1)/2; 0 1 0 BodyDim(2)/2; 0 0 1 0; 0 0 0 1];
BC2LTmat(:,:,3) = [-1 0 0 -BodyDim(1)/2; 0 1 0 -BodyDim(2)/2; 0 0 1 0; 0 0 0 1];
BC2LTmat(:,:,4) = [1 0 0 BodyDim(1)/2; 0 1 0 -BodyDim(2)/2; 0 0 1 0; 0 0 0 1];



%% Parameters of Quadruped and Type of Gaits 
%**************************************************************************
% LOOK UP TABLE for Different Gaits
% [phi4 phi3 phi1 phi2 beta]
% phi = Phase of Legs. Which leg will move at what (normalized) time

LookUp = [    0 0.5 0.75 0.25 0.75;... % Crawl Gait Beta 0.75
              0.9 0.4 0.7 0.2 0.8;...  % Crawl Gait Beta 0.8  
              0.5 0 0.5 0 0.5;...      % Pace Gait
              0 0 0.5 0.5 0.5];        % Gallop Gait
%Gaits:
% 1 - Crawl (beta 0.0.75)
% 2 - Crawl (beta 0.8)
% 3 - Pace
% 4 - Gallop
gait = 1;
%**************************************************************************

%**************************************************************************
CT = 4; %Cycle Time
Fc = 0.07; %foot clearance or foot hieght to be lifted
Hb = 0.09; %hieght of body;
Ls = 0.14; %step length
sc = 0.15; % Side Clearence 
%**************************************************************************
G2BTmat = eye(4);
G2BTmat(3,4) = Hb;
%**************************************************************************
% Velocity of Quadruped
% V = Step Length/Cycle Time
velocity = Ls/(LookUp(gait,5)*CT);
display(velocity);
%**************************************************************************
%% Generating Gait Specific Trajectory 

no_steps = 300; % Number of Steps (Computation)
ang = zeros(4,3,no_steps);
% CRAWL:
% 1 for Y Crawl
% 2 for X Crawl
crawl = 1;

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
   ang(:,:,i) = leg_Yangles(:,:,2);
   y_positions = fwdkinematics(leg_Yangles(:,:,2));
   configs_exp(:,:,i) = ly_footpos;      % Experimental Matrix to check z coordinates
   
   world_Ypos = local2world(y_positions);
   y_configs(:,:,:,i) = world_Ypos + ((j-1)*velocity*CT*y_cont);
   
end
end

leg_Yangles(1,:,2) = 0;
leg_Yangles(2,:,2) = 0;
leg_Yangles(3,:,2) = 0;
leg_Yangles(4,:,2) = 0;
y_positions = fwdkinematics(leg_Yangles(:,:,2));
world_Ypos = local2world(y_positions);
y_configs(:,:,:,i) = world_Ypos + ((j-1)*velocity*CT*y_cont);
%subplot(1,2,1)

drawRef(y_configs(:,:,:,i), 'Original zero position',['1','2','3','4'], [[0 0 -1]; [0 0 1]; [0 0 -1]; [0 0 1]]);

%transformation of reference
% change legs
leg_Yangles(:,:,2) = leg_Yangles([4, 2, 1, 3],:,2);
leg_Yangles(:,:,2) = leg_Yangles(:,:,2).*[  1	1	1
                                            1   1	1
                                            -1	1	1
                                            -1	1	1];
leg_Yangles(:,:,2) = leg_Yangles(:,:,2)+[   45 0 -90
                                            45 0 -90
                                            -45 0 -90
                                            -45 0 -90];
%     leg_Yangles2(:,:) = leg_Yangles([4, 2, 1, 3],:,2);
%     %leg_Yangles2(:,:) = zeros(4,3)+30;
% 
%     leg_Yangles2(:,:) = leg_Yangles2(:,:).*[    1	    1   1
%                                                 1       1   1
%                                                 -1	    1   1
%                                                 -1	    1   1];
% 
%     leg_Yangles2(:,:) = leg_Yangles2(:,:)+[     45      0   90
%                                                 -45     0   90
%                                                 45      0   90
%                                                 -45     0   90];
                                        
y_positions = fwdkinematics(leg_Yangles(:,:,2));
world_Ypos = local2world(y_positions);
y_configs(:,:,:,i) = world_Ypos + ((j-1)*velocity*CT*y_cont);
%subplot(1,2,2)
figure()
drawRef(y_configs(:,:,:,i), 'new zero position',['2','1','3','4'], [[0 0 1];[0 0 1];[0 0 1];[0 0 1];]);
