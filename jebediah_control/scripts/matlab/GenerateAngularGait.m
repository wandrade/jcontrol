function ang = GenerateAngularGait(direction, walkDistance, bh, plot, steps)
    % Direction 1 for X, -1 for -X, 2 for Y, -2 for -Y
    % Velocity in m/s
    % Distance for linear motion in m
    % Direction to turn: 1 turn, 0 no turn
    %Gaits:
    % 1 - Crawl (beta 0.0.75)
    % 2 - Crawl (beta 0.8)
    % 3 - Pace5
    % 4 - Gallop
    %gait = 4;
    gait = 1;
    turn = 0;
%% Defining Basic Variables and Transformations
global BodyDim;  %%Dimensions of Body
global LinkLengths;  %% Link lengths of Leg
global DHParam;  %% DH Parameters of Legs
global G2BTmat;  %% Ground to Body Conter Transformation Matrix
global BC2LTmat; %% Body Center to Leg Transformation Matrix

BodyDim = [0.140 0.140]; %% Dimensions: X Y (Width Length)
LinkLengths = [0.065; 0.103; 0.163];
DHParam = [[0;0;0], LinkLengths, [90;0;0]]; % DH Parameters d a alpha


range_factor = 3.5;
clear_factor = 1.03;
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

%**************************************************************************

%**************************************************************************
CT = 0.001; %Cycle Time
Fc = 0.08; %foot clearance or foot hieght to be lifted
Hb = bh; %hieght of body;
Ls = 0.052; %step length
sc = 0.1; % Side Clearence 
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

no_steps = steps; % Number of Steps (Computation)
ang = zeros(1,4,3);
ang2 = zeros(1,4,3);
% There will be 2 types of trajectories; Linear and Rotation. 
% LTrajectory will generate set of points (=no_steps) in space for ...
% linear motion.
% Output matrix is 4x3x100 where each matrix contains x, y and z position
% for all the 4 legs for all 100 interations.

configs_exp = zeros(4,3);
x_configs = zeros(4,3,4);
plotVec = zeros(4,3,4);
x_cont = zeros(4,3,4);  %Continuously move in Y direction for no_cycles times
x_cont(:,1,:) = 1;
txy_configs = zeros(4,3,4);
X_Trajectory = zeros(4,3,no_steps);

j = 0;
walk = 1;

maxC = -180*ones(4,1);
minC = 180*ones(4,1);
while walk
    j = j+1;
    LinearX_Trajectory = LXTrajectory(LookUp(gait,:), Fc, Ls, Hb, no_steps, sc);
    for i=1:no_steps

        lx_footpos = [  -LinearX_Trajectory(1,1,i)  0.15 LinearX_Trajectory(1,3,i);...
             LinearX_Trajectory(2,1,i)  0.15 LinearX_Trajectory(2,3,i);...
            -LinearX_Trajectory(3,1,i) -0.15 LinearX_Trajectory(3,3,i);...
             LinearX_Trajectory(4,1,i) -0.15 LinearX_Trajectory(4,3,i)];

        X_Trajectory(:,:,i) = lx_footpos;              

        G2BTmat(1,4) =  ((i*Ls)/(LookUp(gait,5)*no_steps));

        y_cont = zeros(4,3,4);  %Continuously move in Y direction for no_cycles times
        y_cont(:,2,:) = 1;

        leg_Xangles = inverseKinematics(lx_footpos);      
        for l = 1:4
            if maxC(l) < leg_Xangles(l,1,2)
                maxC(l) = leg_Xangles(l,1,2);
            end
            if minC(l) > leg_Xangles(l,1,2)
                minC(l) = leg_Xangles(l,1,2);
            end
        end
        
        x_positions = fwdkinematics(leg_Xangles(:,:,2));
        configs_exp(:,:,i) = lx_footpos;      % Experimental Matrix to check z coordinates

        world_Xpos = local2world(x_positions);
        x_configs(:,:,:,i) =  world_Xpos + ((j-1)*velocity*CT*x_cont) + (velocity*CT*y_cont);
        
        step(:,:) = leg_Xangles([4, 2, 1, 3],:,2);
        %leg_Yangles2(:,:) = zeros(4,3)+30;

        step(:,:) = step(:,:).*[    1	    1   1
                                                    1       1   1
                                                    -1	    1   1
                                                    -1	    1   1];

        step(:,:) = step(:,:)+[     45      0   90
                                                    -45     0   90
                                                    45      0   90
                                                    -45     0   90];   
        plotVec(:,:,:,(i+(j-1)*no_steps)) = x_configs(:,:,:,i);
    end
    dist = (x_configs(1,1,1,i)+x_configs(1,1,4,i))/2;
    if dist >= walkDistance
    	walk = 0;
    	break
    end
end
%% simulate walk
j = 0;
walk = 1;
while walk
    j = j+1;
    LinearX_Trajectory = LXTrajectory(LookUp(gait,:), Fc, Ls, Hb, no_steps, sc);
    for i=1:no_steps

        lx_footpos = [  -LinearX_Trajectory(1,1,i)  0.15 LinearX_Trajectory(1,3,i);...
             LinearX_Trajectory(2,1,i)  0.15 LinearX_Trajectory(2,3,i);...
            -LinearX_Trajectory(3,1,i) -0.15 LinearX_Trajectory(3,3,i);...
             LinearX_Trajectory(4,1,i) -0.15 LinearX_Trajectory(4,3,i)];

        X_Trajectory(:,:,i) = lx_footpos;              

        G2BTmat(1,4) =  ((i*Ls)/(LookUp(gait,5)*no_steps));

        y_cont = zeros(4,3,4);  %Continuously move in Y direction for no_cycles times
        y_cont(:,2,:) = 1;

        leg_Xangles = inverseKinematics(lx_footpos);      
        
        % Widden the steps
            leg_Xangles(1,1,2) = mapfun(leg_Xangles(1,1,2), minC(1), maxC(1), minC(1)/range_factor, maxC(1)/clear_factor);
            leg_Xangles(2,1,2) = mapfun(leg_Xangles(2,1,2), minC(2), maxC(2), minC(2)/range_factor, maxC(2)/clear_factor);
            leg_Xangles(3,1,2) = mapfun(leg_Xangles(3,1,2), minC(3), maxC(3), minC(3)/clear_factor, maxC(3)/range_factor);
            leg_Xangles(4,1,2) = mapfun(leg_Xangles(4,1,2), minC(4), maxC(4), minC(4)/clear_factor, maxC(4)/range_factor);
        x_positions = fwdkinematics(leg_Xangles(:,:,2));
        configs_exp(:,:,i) = lx_footpos;      % Experimental Matrix to check z coordinates

        world_Xpos = local2world(x_positions);
        x_configs(:,:,:,i) =  world_Xpos + ((j-1)*velocity*CT*x_cont) + (velocity*CT*y_cont);
        
        step(:,:) = leg_Xangles([4, 1, 2, 3],:,2);

        step(:,:) = step(:,:).*[    1	    1   1
                                                    1       1   1
                                                    -1	    1   1
                                                    -1	    1   1];

        step(:,:) = step(:,:)+[     45      0   90
                                                    -45     0   90
                                                    45      0   90
                                                    -45     0   90];

        ang((i+(j-1)*no_steps),:,:) = step(:,:);       
        
        plotVec(:,:,:,(i+(j-1)*no_steps)) = x_configs(:,:,:,i);
    end
    dist = (x_configs(1,1,1,i)+x_configs(1,1,4,i))/2;
    if dist >= walkDistance
    	walk = 0;
    	break
    end
end
% ang mapeado em (passo,pata,junta)

%% Plot
if plot
    clf
    for i = 1:plotL
        draw(plotVec(:,:,:,i));
        pause(0.0005);
    end
    if turn
        for i = 1:length(ang2)
            draw(txy_configs(:,:,:,i));
            pause(0.0005);

        end
    end
end

if direction > 0
    aux = ang;
    ang = zeros(1,4,3);
    L=length(aux);
    for i=1:L
        index = 1+L-i;
        ang(i,:,:) = aux(index,:,:);   
    end
end
