function [stm] = draw( positions )
global LinkLengths
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
plot3(  positions(:,1,1),positions(:,2,1),positions(:,3,1),'- .r',...
        positions(:,1,2),positions(:,2,2),positions(:,3,2),'- .r',...
        positions(:,1,3),positions(:,2,3),positions(:,3,3),'- .r',...
        positions(:,1,4),positions(:,2,4),positions(:,3,4),'- .r',...
        body(:,1),body(:,2),body(:,3),'- b.',...
        support_polygon(:,1),support_polygon(:,2),support_polygon(:,3),'-- k.',...
        com(1),com(2),com(3),'- k.',...
        'LineWidth',2)
    axis equal;
hold on

scatter3(com(1),com(2) ,com(3) ,'black', 'filled')
scatter3(com(1),com(2), 0.08,'black', 'filled')

quiver3(com(1),com(2), 0.08, 0.15, 0, 0,'color',[0.35,0.35,0.35],'LineWidth',2)
display (LinkLengths)

hold off
grid on;
xlabel('X');
ylabel('Y');
zlabel('Z');
%xlim([-0.3 1]);
%ylim([-0.3 1]);
%zlim([-0 0.40]);
title('Isometric View');
view([50 50]);

end

