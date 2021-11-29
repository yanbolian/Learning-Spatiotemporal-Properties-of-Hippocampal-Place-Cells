function [Vs, RDs, positions, ts, route] = generate_virtual_rat_trajectory(...
            period, env, rat_run, current_route, figure_on)
% Vs: returned speed in m/s
% HDs: returned head direction / running direction; in degrees
% positions: Nt*2; [x y] for the period
% route: information about the trajectory
% 
% peirod: period of running; in seconds
% env: environment characteristics, size and discretized resolution
% rat_run: parameters of running:
% current_route: information of current route;
% figure_on: to display the trajector or not


delta_x = env.x_size / (env.Nx-1);
delta_y = env.y_size / (env.Ny-1);

% Latest state of the route
xt_1 = current_route.latest_x;
yt_1 = current_route.latest_y;
wt_1 = deg2rad(current_route.latest_w);
vt_1 = current_route.latest_v;

dt = current_route.dt;
Nt = period / dt;
ts = (0 : dt : (Nt-1)*dt)';
Vs = zeros(Nt,1);
RDs = zeros(Nt,1);
positions = zeros(Nt,2);

max_step_stuck = 0;
for nt = 1 : Nt
      
    % Update the speed, vt, by Ornstein-Uhlenbeck process
    vt = 0;
    while vt <= 0 % Make sure that vt is not negative
        dw_v = sqrt(dt) * randn;
        dv = rat_run.theta_v * (rat_run.mu_v - vt_1) *dt + rat_run.sigma_v * dw_v;
        vt = vt_1 + dv;
    end
    
    % Update position and make sure the latest position is in bound
    n_step_stuck = 0;
    in_bound = 0;
    while in_bound == 0
        n_step_stuck = n_step_stuck + 1;
        
        % If the object is near the wall, run parallel to the wall
        nearest_wall = find_nearest_wall(xt_1, yt_1, env.x_size, env.y_size, current_route.bound_buffer);
        wt_1 = turn_near_wall(xt_1, yt_1, env.x_size, env.y_size, nearest_wall, wt_1, current_route.bound_buffer);
        
        dw = sqrt(dt) * randn;
        wt = wt_1 + dw;
        theta_t = rat_run.sigma_theta * wt;
        dx = dt * vt * cos(theta_t);
        dy = dt * vt * sin(theta_t);
                
        xt = xt_1 + dx;
        yt = yt_1 + dy;
        
        in_bound = is_in_bound(xt, yt, env.x_size, env.y_size);
%         pause
    end    
    max_step_stuck = max(max_step_stuck, n_step_stuck);
   
    % discretized trajectories; used for plotting
    current_route.Trajectory(round(yt/delta_y)+1, round(xt/delta_x)+1) = current_route.Trajectory(round(yt/delta_y)+1, round(xt/delta_x)+1) + 1;
    
    RDs(nt) = rad2deg(mod(theta_t, 2*pi));
    Vs(nt) = vt;
%     positions(nt,:) = [xt_1 yt_1];
    positions(nt,:) = [xt yt];
    
    % Update the states at time t-1
    xt_1 = xt;
    yt_1 = yt;
    wt_1 = mod(wt, 2*pi);
    vt_1 = vt;
    
    if(mod(nt*dt, 10)==0)
        fprintf('%5d s\n',nt*dt);
    end
end
current_route.latest_w = RDs(nt); % Update latest direction of the trajectory
current_route.latest_v = vt; % Update latest direction of the trajectory
current_route.latest_x = xt; % Update latest direction of the trajectory
current_route.latest_y = yt; % Update latest direction of the trajectory
current_route.T = current_route.T + period;

route = current_route;
% max_step_stuck

if figure_on == 1
figure();
subplot(221)
plot(dt:dt:period, RDs, 'r-');
ylim([0 360]);
ylabel('Head direction (\circ)')
xlabel('Time (s)')
subplot(223)
plot(dt:dt:period, Vs, 'r-');
ylabel('Speed (m/s)');
xlabel('Time (s)')
subplot(122)
imagesc(route.Trajectory);
axis image
% set(gca, 'YDir', 'normal')
set(gca, 'XTick', [1,env.Nx/2,env.Nx], 'XTickLabels', [0,env.x_size/2,env.x_size])
set(gca, 'YTick', [1,env.Ny/2,env.Ny], 'YTickLabels', [0,env.y_size/2,env.y_size])
end