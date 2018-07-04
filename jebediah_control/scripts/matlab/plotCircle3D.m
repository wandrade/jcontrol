function plotCircle3D(center,normal,radius, range, Ccolor)

theta=range(1):0.005:range(2);
v=null(normal);
points=repmat(center',1,size(theta,2))+radius*(v(:,1)*cos(theta)+v(:,2)*sin(theta));
plot3(points(1,:),points(2,:),points(3,:),'-', 'linewidth',1, 'color',Ccolor);

mArrow3(points(:,end-100),points(:,end), 'color', Ccolor, 'stemWidth', 0.2); 
end