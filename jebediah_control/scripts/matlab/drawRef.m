function [stm] = drawRef(positions, titleS, n, normal)
global LinkLengths;
% This function calculates the stability margin and plots the position on a
% 3D Plot.

body = [positions(1,1,1),positions(1,2,1),positions(1,3,1);
        positions(1,1,2),positions(1,2,2),positions(1,3,2);
        positions(1,1,4),positions(1,2,4),positions(1,3,4);
        positions(1,1,3),positions(1,2,3),positions(1,3,3);
        positions(1,1,1),positions(1,2,1),positions(1,3,1)];
    
support_polygon = zeros(1,3);
ind = 1;
min_s = 0.0001;
if(positions(4,3,1)<min_s)
        support_polygon(ind,:) = [positions(4,1,1),positions(4,2,1),0];
        ind = ind+1;
end
if(positions(4,3,2)<min_s)
        support_polygon(ind,:) = [positions(4,1,2),positions(4,2,2),0];
        ind = ind+1;
end
if(positions(4,3,4)<min_s)
        support_polygon(ind,:) = [positions(4,1,4),positions(4,2,4),0];
        ind = ind+1;
end
if(positions(4,3,3)<min_s)
        support_polygon(ind,:) = [positions(4,1,3),positions(4,2,3),0];
end

support_polygon(end+1,:) = support_polygon(1,:);
com = [(positions(1,1,1)+positions(1,1,4))/2,(positions(1,2,1)+positions(1,2,4))/2,0];
stm = getStablityMargin(support_polygon,com);
hold on

% plot front vector
quiver3(com(1),com(2),LinkLengths(2), 0, 0.2, 0,'color',[0.35,0.35,0.35],'LineWidth',2)
plotCircle3D([com(1) com(2) LinkLengths(2)],[0 0 1], 0.02, [pi, 2*pi],[1 0 0])

% Links
plot3(  positions(:,1,1),positions(:,2,1),positions(:,3,1),'-','color',[0.35,0.35,0.35],'LineWidth',2)
plot3(  positions(:,1,2),positions(:,2,2),positions(:,3,2),'-','color',[0.35,0.35,0.35],'LineWidth',2)
plot3(  positions(:,1,3),positions(:,2,3),positions(:,3,3),'-','color',[0.35,0.35,0.35],'LineWidth',2)
plot3(  positions(:,1,4),positions(:,2,4),positions(:,3,4),'-','color',[0.35,0.35,0.35],'LineWidth',2)
plot3(  body(:,1),body(:,2),body(:,3),'-.k','LineWidth',2)
plot3(  com(1),com(2),com(3),'- w.','LineWidth',2)


% plot joints
scatter3(positions(1:3,1,1),positions(1:3,2,1),positions(1:3,3,1),'red', 'filled')
plotCircle3D(positions(1,:,1),normal(1,:), 0.02, [pi, 2*pi],[0.5 0.3 1])

scatter3(positions(1:3,1,2),positions(1:3,2,2),positions(1:3,3,2),'red', 'filled')
plotCircle3D(positions(1,:,2),normal(2,:), 0.02, [pi, 2*pi],[0.5 0.3 1])

scatter3(positions(1:3,1,3),positions(1:3,2,3),positions(1:3,3,3),'red', 'filled')
plotCircle3D(positions(1,:,3),normal(3,:), 0.02, [pi, 2*pi],[0.5 0.3 1])

scatter3(positions(1:3,1,4),positions(1:3,2,4),positions(1:3,3,4),'red', 'filled')
plotCircle3D(positions(1,:,4),normal(4,:), 0.02, [pi, 2*pi],[0.5 0.3 1])


% leg number
text(positions(1,1,1)+0.015,positions(1,2,1)+0.015,positions(1,3,1)+0.015,n(1))
text(positions(1,1,2)+0.015,positions(1,2,2)+0.015,positions(1,3,2)+0.015,n(2))
text(positions(1,1,3)+0.015,positions(1,2,3)+0.015,positions(1,3,3)+0.015,n(3))
text(positions(1,1,4)+0.015,positions(1,2,4)+0.015,positions(1,3,4)+0.015,n(4))

axis equal;
grid on;
%xlim([-0.4 0.4]);
%ylim([-0.2 0.6]);
%zlim([-0 0.1]);
title(titleS);
view([23 40]);

set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
set(gca,'zticklabel',[])
%xlabel('X');
%ylabel('Y');
%zlabel('Z');



hold off
end
