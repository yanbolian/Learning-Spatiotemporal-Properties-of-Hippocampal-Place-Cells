function initial_route = reset_route(route)

route.Trajectory = zeros(size(route.Trajectory)); % Ny rows (y axis) and Nx columns (x axis)
route.T = 0;
route.latest_w = 0; % Initial moving direction
route.latest_v = 0; % Initial running speed
route.latest_x = route.x_size/2; % Initial x position
route.latest_y = route.y_size/2; % Initial y position

initial_route = route;