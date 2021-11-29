%% This script plots figures in the manuscript
% Author: Yanbo Lian
% The result datasets used in the paper cannot be uploaded to Github due to its size.
% You are welcome to contact yanbo.lian@unimelb.edu.au to request the data

%% Figure 6 in the manuscript (the figure in Methods section)
clear
close all
clc
addpath('functions')

random_seed = 3;
rng(random_seed)

% define values of grid parameters of both spatial and temporal properties
grid_cell_2D.spacing = 0.5;
grid_cell_2D.orientation = 0; % in degrees
grid_cell_2D.phase = [0 0];
grid_cell_2D.amplitude_scale = 1;
grid_cell_2D.amplitude_std = 0.2;
grid_cell_2D.k_phi = 2;
grid_cell_2D.phi_0 = 320; % in degrees
grid_cell_2D.delta_phi = 300; % phase changes

env.x_size = 1;
env.y_size = 1;
env.Nx = 400;
env.Ny = 400;

% generate profile of the 2D grid cell
[grid_cell_2D.vertices, grid_cell_2D.radii, grid_cell_2D.amplitudes, grid_cell_2D.G] = generate_2D_grid_fields(...
    grid_cell_2D.spacing, grid_cell_2D.orientation, grid_cell_2D.phase, grid_cell_2D.amplitude_std, grid_cell_2D.amplitude_scale, env);

% return a discretized grid field and plot it
display_2D_grid_cells(grid_cell_2D, env);

%%%%%%%%%%%%%%%%%%%%%% Figure 6A in the manuscript %%%%%%%%%%%%%%%%%%%%%
% Plot example grid cell
figure(1); close
fig = figure(1);
fig.set( 'name', '6A-example grid cell', 'units','centimeters', 'Position',[15 15 6 4.5]);
display_matrix(reshape(grid_cell_2D.G, numel(grid_cell_2D.G),1));
colormap(jet_modified);
h = colorbar;
h.Limits = [0 1];
h.Ticks = [0 0.1 1];
h.TickLabels = {'0', '10%', 'Max'};
h.FontSize = 9.5;

%%%%%%%%%%%%%%%%%%%%%% Figure 6B in the manuscript %%%%%%%%%%%%%%%%%%%%%
% test different values of k_phi

figure(2); close
fig = figure(2);
fig.set( 'name', '6B-phase modulation', 'units','centimeters', 'Position',[15 15 8 6]);

delta_t = 0.001; % 1 ms
ts = 0 : delta_t : 1;
i = 0;
font_size = 12;

for k_phi = [0 1 3]
    i = i + 1;
    
    grid_cell_2D.k_phi = k_phi;
    grid_network.cells(1) = grid_cell_2D;
    grid_network.num_cells = length(grid_network.cells);
    
    rc = [0.75 0.4];
    RD = 0;
    responses = compute_2D_grid_cell_spatiotemporal_response(rc, RD, ts, grid_cell_2D);
    figure(2)    
    plot(ts, responses/max(responses(:)), 'linewidth', 2);
    ylim([0,1.15]);
    hold on;
    labels{i} = ['$k_\phi$=' num2str(k_phi)];

end
[hh,icons,plots,txt]  = legend(labels, 'Orientation','horizontal', 'interpreter', 'latex');
p1 = icons(1).Position;
icons(1).Position = [0.16 p1(2) 0];
icons(4).XData = [0.05 0.15];
icons(1).FontSize = 10;

icons(2).Position = [0.46 p1(2) 0];
icons(6).XData = [0.35 0.45];
icons(2).FontSize = 10;

icons(3).Position = [0.76 p1(2) 0];
icons(8).XData = [0.65 0.75];
icons(3).FontSize = 10;
legend boxoff

xlabel('time (s)');
ylabel('phase modulation')
xlim([0 0.5])
ylim([0,1.2]);
set(gca, 'XTick', 0:0.1:0.5, 'XTickLabels', {'0','0.1','0.2','0.3','0.4','0.5'});
set(gca,'FontSize',font_size)

%%%%%%%%%%%%%%%%%%%%%% Figure 6C in the manuscript %%%%%%%%%%%%%%%%%%%%%
% Test different locations (different r)

figure(3); close
font_size = 12;
delta_t = 0.001; % 1 ms
ts = 0 : delta_t : 1;
k_phi = 2;
grid_cell_2D.k_phi = k_phi;

radius = grid_cell_2D.radii(1);
X = rc(1)-0.15 : 0.05 : rc(1)+0.15;

i_label = 0;
labels={};
colors = [];
phis = [];

for x = X
    i_label = i_label + 1;
    r = [x, rc(2)];

    RD = 0;
    responses = compute_2D_grid_cell_spatiotemporal_response(r, RD, ts, grid_cell_2D);
    
    fig = figure(3);
    fig.set( 'name', '6C-spatio-temporal responses at different positions over 0.5s', 'units','centimeters', 'Position',[15 15 8 6]);

    fig_temp = plot(ts, responses, 'linewidth', 2);
    colors = [colors; fig_temp.Color];
    hold on;
    
    labels{i_label} = ['r=(' num2str(r(1)) ',' num2str(r(2)) ')'];
        
    [params, fit_parameters] = fit_spatiotemporal(responses, ts);
    phis = [phis rad2deg(params(3))]
end
% legend(labels)
set(gca,'FontSize',font_size)
xlabel('time (s)');
ylabel('response')
xlim([0 0.5])
ylim([0,0.7]);
set(gca, 'XTick', 0:0.1:0.5, 'XTickLabels', {'0','0.1','0.2','0.3','0.4','0.5'});
% set(gca, 'YTick', [0 1], 'YTickLabels', {'0', 'Max'});
               
%%%%%%%%%%%%%%%%%%%%%% Figure 6D in the manuscript %%%%%%%%%%%%%%%%%%%%%
figure(4); close;
fig = figure(4);
fig.set( 'name', '6D-Phase vs. Position', 'units','centimeters', 'Position',[15 15 5 5]);
plot(X, phis, 'marker', '.', 'markersize', 0.01, 'linestyle', '--', 'linewidth', 1, 'color', 'k');
hold on
scatter(X, phis, 20, colors, 'filled');
for i = 1 : 3
    r = [X(i), rc(2)];
    text(X(i)+0.03,phis(i), ['(' num2str(r(1)) ',' num2str(r(2)) ')'],'fontsize',8)
end
i = 4;
r = [X(i), rc(2)];
text(X(i)+0.015,phis(i), ['(' num2str(r(1)) ',' num2str(r(2)) ')'],'fontsize',8)
for i = 5 : length(X)
    r = [X(i), rc(2)];
    text(X(i)-0.20,phis(i), ['(' num2str(r(1)) ',' num2str(r(2)) ')'],'fontsize',8)
end

xlim([X(1)-0.05 X(end)+0.05])
ylim([0 360])
set(gca, 'XTick', [X(1) X(end)], 'XTickLabels', {X(1), X(end)});
xlabel('x (m)')
ylabel('firing phase, \phi (\circ)') %,'Interpreter','latex');
set(gca,'FontSize',font_size)

%% Figure 1 in the manuscript (spatial properties of Scenario 1)
close all
clear
clc
addpath('functions')
addpath('results')

% The result dataset used in the paper cannot be uploaded to Github due to its size.
% You are welcome to contact yanbo.lian@unimelb.edu.au to request the data
load('Scenario_1.mat')

font_size = 28;

% Select model place cells
fit_errors = [lca.fields_fit(:).fit_error];
% max(fit_errors)

fit_error_threshold = 40; % In percentage;
radius_threshold = 0.05; % Unit: meters.
width_scale = 0;

% Find place cells that meet the criteria: fitting error<15% and radius>5cm
place_cells_index = [];
for i_cell = 1 : lca.num_place_cell
    if lca.fields_fit(i_cell).fit_error < fit_error_threshold ...
            && lca.fields_fit(i_cell).params(4) > radius_threshold ...
            && lca.fields_fit(i_cell).params(2) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(2) < env.x_size-width_scale*lca.fields_fit(i_cell).params(4) ...
            && lca.fields_fit(i_cell).params(3) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(3) < env.y_size-width_scale*lca.fields_fit(i_cell).params(4) ...
        place_cells_index = [place_cells_index i_cell];
    end
end

%%%%%%%%%%%%%%%%%%%%%% Figure 1A in the manuscript %%%%%%%%%%%%%%%%%%%%%
disp = lca.place_field_recovered(:, place_cells_index);
fig = figure();
fig.set( 'name',['1A-' num2str(length(place_cells_index)) ' place cells'],'units','normalized','position',[0.1 0.1 0.45 0.7] )
display_matrix(disp, 3);
colormap(jet_modified)
h = colorbar;
h.Limits = [0 1];
h.Ticks = [0 0.1 1];
h.TickLabels = {'0', '10%', 'Max'};
h.FontSize = 28;

%%%%%%%%%%%%%%%%%%%%%% Figure 1B in the manuscript %%%%%%%%%%%%%%%%%%%%%
Nx = 80;
Ny = 80;
delta_x = env.x_size / (Nx - 1);
delta_y = env.y_size / (Ny - 1);
place_fields_centers = zeros(Nx, Ny);
for i_cell = place_cells_index
    i_x = min(max(1 + round(lca.fields_fit(i_cell).params(2)/delta_x), 1),Nx);
    i_y = min(max(1 + round(lca.fields_fit(i_cell).params(3)/delta_y), 1),Ny);
    place_fields_centers(i_x, i_y) = 1;
end
fig = figure();
fig.set( 'name', '1B-centres of all learnt place cells');
imagesc(place_fields_centers);
axis equal
colormap(gray)
axis image
% set(gca, 'YDir', 'normal')
set(gca, 'XTick', [1,Nx/2,Nx], 'XTickLabels', {'0', [num2str(env.x_size/2) ' m'],[num2str(env.x_size) ' m']})
set(gca, 'YTick', [1,Ny/2,Ny], 'YTickLabels', {'0', [num2str(env.y_size/2) ' m'],[num2str(env.y_size) ' m']})
set(gca, 'FontSize', font_size);

radiuses = zeros(length(place_cells_index), 1);
for i_cell = 1 : length(place_cells_index)
    radiuses(i_cell) = lca.fields_fit( place_cells_index(i_cell) ).params(4);
end
fprintf(['mean radius: ' num2str(mean(100*radiuses)) ', std radius: ' num2str(std(100*radiuses)) '\n'])


%% Figure 2 in the manuscript (tenmporal properties of Scenario 1)
close all
clear
clc
addpath('functions')
addpath('results')

% The result dataset used in the paper cannot be uploaded to Github due to its size.
% You are welcome to contact yanbo.lian@unimelb.edu.au to request the data
load('Scenario_1.mat')

i_place_cell = 3; rc = [0.65 0.75]; % no weakly spatial cells; scenario 1
ts = 0 : 0.001 : 1;
Nt = length(ts);
params = [];
font_size = 12;

RD = 0;

% Initialize membrane potentials and firing rates to zero
U_place = zeros(lca.num_place_cell, Nt); % Membrane potentials of M neurons for batch_size images
S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images
[S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
        weak_network, grid_network, lca, rc, RD, ts, env, S_place, U_place);

[params, fit_parameters] = fit_spatiotemporal(S_place(i_place_cell,:), ts);
phi = rad2deg(params(3))

fun_spatiotemporal_response = @(params, ts) ...
    1 * exp(params(1)*(cos(2*pi*params(2)*ts-params(3))-1));

spatiotemporal_response_fit = params(4) * fun_spatiotemporal_response(params(1:3), ts);

%%%%%%%%%%%%%%%%%%%%%% Figure 2A in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig=figure(1); clf
fig.set( 'name', '2A-example learnt place cell', 'units','centimeters', 'Position',[15 15 6 4.5]);
display_matrix(lca.place_field_recovered(:,i_place_cell), 10); colormap(jet_modified);
h = colorbar;
h.Limits = [0 1];
h.Ticks = [0 0.1 1];
h.TickLabels = {'0', '10%', 'Max'};
h.FontSize = 9.5;

%%%%%%%%%%%%%%%%%%%%%% Figure 2B in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure(2);
fig.set( 'name', '2B-response at center', 'units','centimeters', 'Position',[15 15 8 6]);
plot(ts, S_place(i_place_cell,:)', 'linewidth', 2);
hold on;
plot(ts, spatiotemporal_response_fit, 'linestyle', '--', 'linewidth', 3);
legend({' simulated data',' fitted data'})
xlabel('time (s)');
ylabel('response')
xlim([0 0.5])
ylim([0,3.8]);
set(gca, 'XTick', 0:0.1:0.5, 'XTickLabels', {'0','0.1','0.2','0.3','0.4','0.5'});
set(gca,'FontSize',font_size)

%%%%%%%%%%%%%%%%%%%%%% Figure 2C in the manuscript %%%%%%%%%%%%%%%%%%%%%
% Test the response of a place cell at one location over 1s: from left to right

i_label = 0;
labels={};
phis = [];
font_size = 12;
X = rc(1)-0.15 : 0.05 : rc(1)+0.15;

colors = [];
for x = X
    i_label = i_label + 1;
    r = [x, rc(2)];

    RD = 0; % left to right
    
    % Initialisation for place cells
    U_place = zeros(lca.num_place_cell, Nt); % Membrane potentials of M neurons for batch_size images
    S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images
    [~, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
        weak_network, grid_network, lca, r, RD, ts, env, S_place, U_place);
    
    fig = figure(3);
    fig.set( 'name', '2C', 'units','centimeters', 'Position',[15 15 8 6]);
    fig_temp = plot(ts, S_place(i_place_cell,:)', 'linewidth', 2);
    colors = [colors; fig_temp.Color];
    hold on
    
    labels{i_label} = ['r=(' num2str(r(1)) ',' num2str(r(2)) ')'];
    
    [params, fit_parameters] = fit_spatiotemporal(S_place(i_place_cell,:), ts);
    phis = [phis rad2deg(params(3))]
end
% legend(labels)
set(gca,'FontSize',font_size)
xlabel('time (s)');
ylabel('response')
xlim([0 0.5])
ylim([0,3]);
set(gca, 'XTick', 0:0.1:0.5, 'XTickLabels', {'0','0.1','0.2','0.3','0.4','0.5'});

%%%%%%%%%%%%%%%%%%%%%% Figure 2D in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure(4);
fig.set( 'name', '3D-Phase vs. Position', 'units','centimeters', 'Position',[15 15 5 5]);
plot(X, phis, 'marker', '.', 'markersize', 0.01, 'linestyle', '--', 'linewidth', 1, 'color', 'k');
hold on
scatter(X, phis, 20, colors, 'filled');
for i = 1 : 3
    r = [X(i), rc(2)];
    text(X(i)+0.03,phis(i), ['(' num2str(r(1)) ',' num2str(r(2)) ')'],'fontsize',8)
end
i = 4;
r = [X(i), rc(2)];
text(X(i)+0.015,phis(i), ['(' num2str(r(1)) ',' num2str(r(2)) ')'],'fontsize',8)
for i = 5 : length(X)
    r = [X(i), rc(2)];
    text(X(i)-0.20,phis(i), ['(' num2str(r(1)) ',' num2str(r(2)) ')'],'fontsize',8)
end

xlim([X(1)-0.05 X(end)+0.05])
ylim([0 360])
set(gca, 'XTick', [X(1) X(end)], 'XTickLabels', {X(1), X(end)});
xlabel('x (m)')
ylabel('firing phase (\circ)')
set(gca,'FontSize',font_size)


%%%%%%%%%%%%%%%%%%%%% Figure 2E in the manuscript %%%%%%%%%%%%%%%%%%%%%
% Test the response of a place cell at one location over 1s: a curved trajectory
random_seed = 14;
rng(random_seed)

ts = 0 : 0.001 : 1;
Nt = length(ts);
% i_place_cell = 81; rc = [0.25 0.5]; % 76th of place_cells_index; 0.1 weak400; scenario 2 3
i_place_cell = 3; rc = [0.65 0.75]; % no weakly spatial cells; scenario 1

% Given the centre, generate a curved trajectory:
% 1. choose a starting point: left side of the place field.
R = lca.fields_fit(i_place_cell).params(4);
r_start = [rc(1)-R, rc(2)];

% 3. Stop when the animal is about to leave the place field
% 4. Make sure the trajectory is not too short. Otherwise, re-do the process

% Parameters of running trajectory: same as traing and testing data
rat_run_temp.theta_v = 3; % Large theta_v leads to faster convergence to mean speed: mu_v
rat_run_temp.sigma_v = 0.1; % If sigma_v=0, it does not depend on Wiener process
rat_run_temp.mu_v = 0.3; % m/s; long-term mean speed;
rat_run_temp.sigma_theta = 1; % 0.7 is used in D'Albis et al. 2017; values other than 1 will will make it more difficult near walls

% 2. choose a starting running direction: toward right
route_temp.dt = 0.01; % The resolution of 10 ms; 1 ms also works
route_temp.bound_buffer = 0.01; % 2 cm
route_temp.Trajectory = zeros(env.Ny, env.Nx); % Ny rows (y axis) and Nx columns (x axis)
route_temp.T = 0;
route_temp.latest_w = 0; % Initial moving direction
route_temp.latest_v = 0.25; % Initial running speed
route_temp.latest_x = r_start(1); % Initial x position
route_temp.latest_y = r_start(2); % Initial y position
route_temp.x_size = env.x_size;
route_temp.y_size = env.y_size;

Vs_temp = route_temp.latest_v;
RDs_temp = route_temp.latest_w;
positions_temp = r_start;

% Generate trajectory:
for t = 0 : route_temp.dt : 100
    figure_on = 0;
    [v_temp, rd_temp, position_temp, ~, route_temp] = generate_virtual_rat_trajectory(route_temp.dt, env, rat_run_temp, route_temp, figure_on);
%     toc
    if norm(position_temp-rc)>=R
        break;
    end
    positions_temp = [positions_temp; position_temp];
    Vs_temp = [Vs_temp v_temp];
    RDs_temp = [RDs_temp rd_temp];
    
    pause(0.01)
end
fig = figure(5);
fig.set( 'name','2A-curved trajectory','units','centimeters', 'Position',[15 15 6 4.5])
imagesc(route_temp.Trajectory);
axis image
% set(gca, 'YDir', 'normal')
set(gca, 'XTick', [1,env.Nx/2,env.Nx], 'XTickLabels', [0,env.x_size/2,env.x_size])
set(gca, 'YTick', [1,env.Ny/2,env.Ny], 'YTickLabels', [0,env.y_size/2,env.y_size])

N_temp = length(RDs_temp);
num_points = 7;

i_label = 0;
labels={};
phis = [];
PDCDs = [];
colors = [];
font_size = 12;
for i_point = 1 : ceil(N_temp/num_points) : N_temp
    i_label = i_label + 1;
    
    r = positions_temp(i_point,:);
    RD = RDs_temp(i_point); % bottom to top
	
    % Compute the firing phase
    r_rc = complex(rc(1)-r(1),rc(2)-r(2));
    w = angle(r_rc) - deg2rad(RD);
    R_phi = sqrt(R^2 - abs(r_rc)^2*(sin(w))^2);
    delta_r = abs(r_rc)*cos(w);
    r_phi = R_phi - delta_r;
    PDCDs = [PDCDs -delta_r/R_phi];
    
    
    % Initialisation for place cells
    U_place = zeros(lca.num_place_cell, Nt); % Membrane potentials of M neurons for batch_size images
    S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images
    [S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
        weak_network, grid_network, lca, r, RD, ts, env, S_place, U_place);
    
    fig = figure(6);
    fig.set( 'name', '2E-curved trajectory', 'units','centimeters', 'Position',[15 15 8 6]);
    fig_temp = plot(ts, S_place(i_place_cell,:)', 'linewidth', 2);
    colors = [colors; fig_temp.Color];
    hold on
    
    labels{i_label} = ['r=(' num2str(r(1)) ',' num2str(r(2)) ')'];
    
    [params, fit_parameters] = fit_spatiotemporal(S_place(i_place_cell,:), ts);
    phis = [phis rad2deg(params(3))]
end

% legend(labels)
set(gca,'FontSize',font_size)
xlabel('time (s)');
ylabel('response')
xlim([0 0.5])
ylim([0,3]);
set(gca, 'XTick', 0:0.1:0.5, 'XTickLabels', {'0','0.1','0.2','0.3','0.4','0.5'});

%%%%%%%%%%%%%%%%%%%%%% Figure 2F in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure(7);
fig.set( 'name', '2F-Phase vs. Position', 'units','centimeters', 'Position',[15 15 5 5]);
plot(PDCDs, phis, 'marker', '.', 'markersize', 0.01, 'linestyle', '--', 'linewidth', 1, 'color', 'k');
hold on;
scatter(PDCDs, phis, 20, colors, 'filled');
xlim([-1 1])
ylim([0 360])
set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
xlabel('pdcd')
ylabel('firing phase (\circ)')
set(gca,'FontSize',12)
corr2(phis, PDCDs)

%% Figure 3 in the manuscript (temporal properties of the population in Scenario 1)
% Measure theta phase precession of the population using curved strajectory
close all
clear
clc
addpath('functions')
addpath('results')

% The result dataset used in the paper cannot be uploaded to Github due to its size.
% You are welcome to contact yanbo.lian@unimelb.edu.au to request the data
load('Scenario_1.mat')

% Select model place cells
fit_errors = [lca.fields_fit(:).fit_error];

fit_error_threshold = 40; % In percentage;
radius_threshold = 0.05; % Unit: meters.
width_scale = 1;

% Find place cells that meet the criteria: fitting error<15% and radius>5cm and within the environment
place_cells_index = [];
for i_cell = 1 : lca.num_place_cell
    if lca.fields_fit(i_cell).fit_error < fit_error_threshold ...
            && lca.fields_fit(i_cell).params(4) > radius_threshold ...
            && lca.fields_fit(i_cell).params(2) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(2) < env.x_size-width_scale*lca.fields_fit(i_cell).params(4) ...
            && lca.fields_fit(i_cell).params(3) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(3) < env.y_size-width_scale*lca.fields_fit(i_cell).params(4) ...
        place_cells_index = [place_cells_index i_cell];
    end
end

ts = 0 : 0.001 : 1;
Nt = length(ts);
num_points = 30; % number of points sampled on the trajectory
    
% Given the centre, generate a curved trajectory:
% 1. choose a starting point: left side of the place field and running
% toward the right
% 2. Generate free running trajectory
% 3. Stop when the animal is about to leave the place field

% Parameters of running trajectory: same as traing and testing data
rat_run_temp.theta_v = 3; % Large theta_v leads to faster convergence to mean speed: mu_v
rat_run_temp.sigma_v = 0.1; % If sigma_v=0, it does not depend on Wiener process
rat_run_temp.mu_v = 0.3; % m/s; long-term mean speed;
rat_run_temp.sigma_theta = 1; % 0.7 is used in D'Albis et al. 2017; values other than 1 will will make it more difficult near walls

% 2. choose a starting running direction: toward right
route_temp.dt = 0.01; % The resolution of 10 ms; 1 ms also works
route_temp.bound_buffer = 0.01; % 2 cm
route_temp.x_size = env.x_size;
route_temp.y_size = env.y_size;

entry_phis = [];
exit_phis = [];
corrs = [];
PDCD_population = [];
phi_population = [];

for i_cell = 1 : length(place_cells_index)
        
    fig = figure(1); close;
    i_place_cell = place_cells_index(i_cell);
    R = lca.fields_fit(i_place_cell).params(4);
    rc = [lca.fields_fit(i_place_cell).params(2)...
        lca.fields_fit(i_place_cell).params(3)];
%     i_label = 0;
    r_start = [rc(1)-R, rc(2)];
    
    route_temp.Trajectory = zeros(env.Ny, env.Nx); % Ny rows (y axis) and Nx columns (x axis)
    route_temp.T = 0;
    route_temp.latest_w = 0; % Initial moving direction
    route_temp.latest_v = 0.25; % Initial running speed
    route_temp.latest_x = r_start(1); % Initial x position
    route_temp.latest_y = r_start(2); % Initial y position

    Vs_temp = route_temp.latest_v;
    RDs_temp = route_temp.latest_w;
    positions_temp = r_start;
    
    % Generate a running trajectory starting from the left side of the place field
    for t = 0 : route_temp.dt : 100
        figure_on = 0;
        [v_temp, rd_temp, position_temp, ~, route_temp] = generate_virtual_rat_trajectory(route_temp.dt, env, rat_run_temp, route_temp, figure_on);
        %     toc
        if norm(position_temp-rc)>=R
            break;
        end
        positions_temp = [positions_temp; position_temp];
        Vs_temp = [Vs_temp v_temp];
        RDs_temp = [RDs_temp rd_temp];
        
        pause(0.01)
    end
    
    N_temp = length(RDs_temp);
    
    phis = [];
    PDCDs = [];
    colors = [];
    font_size = 12;
    
    for i_point = 1 : round(N_temp/num_points) : N_temp        
        r = positions_temp(i_point,:);
        RD = RDs_temp(i_point); % bottom to top
        
        % Compute the firing phase
        r_rc = complex(rc(1)-r(1),rc(2)-r(2));
        w = angle(r_rc) - deg2rad(RD);
        R_phi = sqrt(R^2 - abs(r_rc)^2*(sin(w))^2);
        delta_r = abs(r_rc)*cos(w);
        r_phi = R_phi - delta_r;
        
        
        % Initialisation for place cells
        U_place = zeros(lca.num_place_cell, Nt); % Membrane potentials of M neurons for batch_size images
        S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images
        [S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
            weak_network, grid_network, lca, r, RD, ts, env, S_place, U_place);
        
        
        fig = figure(1);
        fig.set( 'name', 'phase precession', 'units','centimeters', 'Position',[15 15 8 6]);
        fig_temp = plot(ts, S_place(i_place_cell,:)', 'linewidth', 2);
        colors = [colors; fig_temp.Color];
        hold on
        
        [params, fit_parameters] = fit_spatiotemporal(S_place(i_place_cell,:), ts);
        
        if ~isempty(params)
            PDCDs = [PDCDs -delta_r/R_phi];
            phis = [phis rad2deg(params(3))];
        end
    end
    
    if phis(end)>300
        phis(end) = phis(end) - 360;
    end
    
    fig = figure(2);
    fig.set( 'name', 'Phase vs. Position', 'units','centimeters', 'Position',[15 15 5 5]);
    scatter(PDCDs, phis, 20, colors, 'filled');
    xlim([-1 1])
    ylim([0 360])
    set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
    xlabel('pdcd')
    ylabel('firing phase (\circ)')
    set(gca,'FontSize',12)
    
    i_cell
    corr2(phis, PDCDs)
    
    PDCD_population = [PDCD_population PDCDs];
    phi_population = [phi_population phis];
    corrs(i_cell) = corr2(phis, PDCDs);
    entry_phis(i_cell) = phis(1);
    exit_phis(i_cell) = phis(end);

%     pause
end
close all

% Plot temporal property of the population using curved trajectories
font_size = 12;

%%%%%%%%%%%%%%%%%%%%%% Figure 3A in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure();
fig.set( 'name','3A-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 5.5 4])
histogram(entry_phis, [-60 0 60 120 180 240 300 360 420]);
xlabel('entry phase (\circ)');
ylabel('counts');
ylim([0 65]);
set(gca, 'XTick', [0 180 360], 'XTickLabels', {0, 180, 360});
text(180,50,['N=' num2str(length(place_cells_index))],'fontsize',font_size)
set(gca,'FontSize',font_size);

%%%%%%%%%%%%%%%%%%%%%% Figure 3B in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure();
fig.set( 'name','3B-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 5.5 4])
histogram(exit_phis, [-60 0 60 120 180 240 300 360 420]);xlabel('exit phase (\circ)');
ylabel('counts');
ylim([0 65]);
set(gca, 'XTick', [0 180 360], 'XTickLabels', {0, 180, 360});
text(180,50,['N=' num2str(length(place_cells_index))],'fontsize',font_size)
set(gca,'FontSize',font_size);

%%%%%%%%%%%%%%%%%%%%%% Figure 3C in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure();
fig.set( 'name','3C-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 5.5 4])
histogram(corrs, -1:0.2: 1);
xlabel('corr. coef.');
ylabel('counts');
% xlim([-1 -0.97]);
ylim([0 65]);
set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
text(0,50,['N=' num2str(length(place_cells_index))],'fontsize',font_size)
set(gca,'FontSize',font_size);

%%%%%%%%%%%%%%%%%%%%%% Figure 3D in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure();
fig.set( 'name','3D-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 10 7])
scatter(PDCD_population(PDCD_population>-0.999), phi_population(PDCD_population>-0.999),2, 'filled')
xlabel('pdcd');
ylabel('firing phase (\circ)')
xlim([-1 1]);
ylim([0 360]);
axis square
set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
set(gca, 'YTick', [0 180 360], 'YTickLabels', {0, 180, 360});
set(gca,'FontSize',font_size);
% c = polyfit(PDCD_population(PDCD_population>-0.999),phi_population(PDCD_population>-0.999),1);

%% Figure 4A & B in the manuscript (Scenario 2)
close all
clear
clc
addpath('functions')
addpath('results')

% The result dataset used in the paper cannot be uploaded to Github due to its size.
% You are welcome to contact yanbo.lian@unimelb.edu.au to request the data
load('Scenario_2&3.mat')

font_size = 28;

% Select model place cells
fit_errors = [lca.fields_fit(:).fit_error];
% max(fit_errors)

fit_error_threshold = 40; % In percentage;
radius_threshold = 0.05; % Unit: meters.

width_scale = 0; 

% Find place cells that meet the criteria: fitting error<15% and radius>5cm
place_cells_index = [];
for i_cell = 1 : lca.num_place_cell
    if lca.fields_fit(i_cell).fit_error < fit_error_threshold ...
            && lca.fields_fit(i_cell).params(4) > radius_threshold ...
            && lca.fields_fit(i_cell).params(2) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(2) < env.x_size-width_scale*lca.fields_fit(i_cell).params(4) ...
            && lca.fields_fit(i_cell).params(3) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(3) < env.y_size-width_scale*lca.fields_fit(i_cell).params(4) ...
        place_cells_index = [place_cells_index i_cell];
    end
end

%%%%%%%%%%%%%%%%%%%%%% Figure 4A in the manuscript %%%%%%%%%%%%%%%%%%%%%
disp = lca.place_field_recovered(:, place_cells_index);
fig = figure(1);
fig.set( 'name',['4A (left)-' num2str(length(place_cells_index)) ' place cells'],'units','normalized','position',[0.1 0.1 0.45 0.7] )
display_matrix(disp, 3);
colormap(jet_modified)
h = colorbar;
h.Limits = [0 1];
h.Ticks = [0 0.1 1];
h.TickLabels = {'0', '10%', 'Max'};
h.FontSize = 28;

Nx = 80;
Ny = 80;
delta_x = env.x_size / (Nx - 1);
delta_y = env.y_size / (Ny - 1);
place_fields_centers = zeros(Nx, Ny);
for i_cell = place_cells_index
    i_x = min(max(1 + round(lca.fields_fit(i_cell).params(2)/delta_x), 1),Nx);
    i_y = min(max(1 + round(lca.fields_fit(i_cell).params(3)/delta_y), 1),Ny);
    place_fields_centers(i_x, i_y) = 1;
end
fig = figure(2);
fig.set( 'name', '4A (right)-centres of all learnt place cells');
imagesc(place_fields_centers);
axis equal
colormap(gray)
axis image
% set(gca, 'YDir', 'normal')
set(gca, 'XTick', [1,Nx/2,Nx], 'XTickLabels', {'0', [num2str(env.x_size/2) ' m'],[num2str(env.x_size) ' m']})
set(gca, 'YTick', [1,Ny/2,Ny], 'YTickLabels', {'0', [num2str(env.y_size/2) ' m'],[num2str(env.y_size) ' m']})
set(gca, 'FontSize', font_size);

radiuses = zeros(length(place_cells_index), 1);
for i_cell = 1 : length(place_cells_index)
    radiuses(i_cell) = lca.fields_fit( place_cells_index(i_cell) ).params(4);
end
fprintf(['mean radius: ' num2str(mean(100*radiuses)) ', std radius: ' num2str(std(100*radiuses)) '\n'])


%%%%%%%%%%%%%%%%%%%%%% Figure 4B in the manuscript %%%%%%%%%%%%%%%%%%%%%
i_place_cell = 81; rc = [0.25 0.5]; % 76th of place_cells_index; 0.1 weak400; scenario 2 3
ts = 0 : 0.001 : 1;
Nt = length(ts);
params = [];
font_size = 12;
random_seed = 14;
rng(random_seed)


% Given the centre, generate a curved trajectory:
% 1. choose a starting point: left side of the place field.
R = lca.fields_fit(i_place_cell).params(4);
r_start = [rc(1)-R, rc(2)];

% 3. Stop when the animal is about to leave the place field
% 4. Make sure the trajectory is not too short. Otherwise, re-do the process

% Parameters of running trajectory: same as traing and testing data
rat_run_temp.theta_v = 3; % Large theta_v leads to faster convergence to mean speed: mu_v
rat_run_temp.sigma_v = 0.1; % If sigma_v=0, it does not depend on Wiener process
rat_run_temp.mu_v = 0.3; % m/s; long-term mean speed;
rat_run_temp.sigma_theta = 1; % 0.7 is used in D'Albis et al. 2017; values other than 1 will will make it more difficult near walls

% 2. choose a starting running direction: toward right
route_temp.dt = 0.01; % The resolution of 10 ms; 1 ms also works
route_temp.bound_buffer = 0.01; % 2 cm
route_temp.Trajectory = zeros(env.Ny, env.Nx); % Ny rows (y axis) and Nx columns (x axis)
route_temp.T = 0;
route_temp.latest_w = 0; % Initial moving direction
route_temp.latest_v = 0.25; % Initial running speed
route_temp.latest_x = r_start(1); % Initial x position
route_temp.latest_y = r_start(2); % Initial y position
route_temp.x_size = env.x_size;
route_temp.y_size = env.y_size;

Vs_temp = route_temp.latest_v;
RDs_temp = route_temp.latest_w;
positions_temp = r_start;

% Generate trajectory:
for t = 0 : route_temp.dt : 100
    figure_on = 0;
    [v_temp, rd_temp, position_temp, ~, route_temp] = generate_virtual_rat_trajectory(route_temp.dt, env, rat_run_temp, route_temp, figure_on);
%     toc
    if norm(position_temp-rc)>=R
        break;
    end
    positions_temp = [positions_temp; position_temp];
    Vs_temp = [Vs_temp v_temp];
    RDs_temp = [RDs_temp rd_temp];
    
    pause(0.01)
end

N_temp = length(RDs_temp);
num_points = 7;

i_label = 0;
labels={};
phis = [];
PDCDs = [];
colors = [];
font_size = 12;
for i_point = 1 : ceil(N_temp/num_points) : N_temp
    i_label = i_label + 1;
    
    r = positions_temp(i_point,:);
    RD = RDs_temp(i_point); % bottom to top
	
    % Compute the firing phase
    r_rc = complex(rc(1)-r(1),rc(2)-r(2));
    w = angle(r_rc) - deg2rad(RD);
    R_phi = sqrt(R^2 - abs(r_rc)^2*(sin(w))^2);
    delta_r = abs(r_rc)*cos(w);
    r_phi = R_phi - delta_r;
    PDCDs = [PDCDs -delta_r/R_phi];
    
    % Initialisation for place cells
    U_place = zeros(lca.num_place_cell, Nt); % Membrane potentials of M neurons for batch_size images
    S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images
    [S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
        weak_network, grid_network, lca, r, RD, ts, env, S_place, U_place);
    
    fig = figure(3);
    fig.set( 'name', '4-curved trajectory', 'units','centimeters', 'Position',[15 15 8 6]);
    fig_temp = plot(ts, S_place(i_place_cell,:)', 'linewidth', 2);
    colors = [colors; fig_temp.Color];
    hold on
    
    labels{i_label} = ['r=(' num2str(r(1)) ',' num2str(r(2)) ')'];
    
    [params, fit_parameters] = fit_spatiotemporal(S_place(i_place_cell,:), ts);
    phis = [phis rad2deg(params(3))]
end

fig = figure(3);clf
fig.set( 'name', '4B-Phase vs. Position', 'units','centimeters', 'Position',[15 15 5 5]);
plot(PDCDs, phis, 'marker', '.', 'markersize', 0.01, 'linestyle', '--', 'linewidth', 1, 'color', 'k');
hold on;
scatter(PDCDs, phis, 20, colors, 'filled');
xlim([-1 1])
ylim([0 360])
set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
xlabel('pdcd')
ylabel('firing phase (\circ)')
set(gca,'FontSize',12)
corr2(phis, PDCDs)


%% Figure 4C and D in the manuscript (Temporal properties of the population in Scenario 2)
% Measure theta phase precession of the population using curved strajectory
close all
clear
clc
addpath('functions')
addpath('results')

% The result dataset used in the paper cannot be uploaded to Github due to its size.
% You are welcome to contact yanbo.lian@unimelb.edu.au to request the data
load('Scenario_2&3.mat')

% Select model place cells
fit_errors = [lca.fields_fit(:).fit_error];

fit_error_threshold = 40; % In percentage;
radius_threshold = 0.05; % Unit: meters.
width_scale = 1;

% Find place cells that meet the criteria: fitting error<15% and radius>5cm and within the environment
place_cells_index = [];
for i_cell = 1 : lca.num_place_cell
    if lca.fields_fit(i_cell).fit_error < fit_error_threshold ...
            && lca.fields_fit(i_cell).params(4) > radius_threshold ...
            && lca.fields_fit(i_cell).params(2) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(2) < env.x_size-width_scale*lca.fields_fit(i_cell).params(4) ...
            && lca.fields_fit(i_cell).params(3) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(3) < env.y_size-width_scale*lca.fields_fit(i_cell).params(4) ...
        place_cells_index = [place_cells_index i_cell];
    end
end

ts = 0 : 0.001 : 1;
Nt = length(ts);
num_points = 30; % number of points sampled on the trajectory
    
% Given the centre, generate a curved trajectory:
% 1. choose a starting point: left side of the place field and running
% toward the right
% 2. Generate free running trajectory
% 3. Stop when the animal is about to leave the place field

% Parameters of running trajectory: same as traing and testing data
rat_run_temp.theta_v = 3; % Large theta_v leads to faster convergence to mean speed: mu_v
rat_run_temp.sigma_v = 0.1; % If sigma_v=0, it does not depend on Wiener process
rat_run_temp.mu_v = 0.3; % m/s; long-term mean speed;
rat_run_temp.sigma_theta = 1; % 0.7 is used in D'Albis et al. 2017; values other than 1 will will make it more difficult near walls

% 2. choose a starting running direction: toward right
route_temp.dt = 0.01; % The resolution of 10 ms; 1 ms also works
route_temp.bound_buffer = 0.01; % 2 cm
route_temp.x_size = env.x_size;
route_temp.y_size = env.y_size;

entry_phis = [];
exit_phis = [];
corrs = [];
PDCD_population = [];
phi_population = [];

for i_cell = 1 : length(place_cells_index)
        
    fig = figure(1); close;
    i_place_cell = place_cells_index(i_cell);
    R = lca.fields_fit(i_place_cell).params(4);
    rc = [lca.fields_fit(i_place_cell).params(2)...
        lca.fields_fit(i_place_cell).params(3)];
%     i_label = 0;
    r_start = [rc(1)-R, rc(2)];
    
    route_temp.Trajectory = zeros(env.Ny, env.Nx); % Ny rows (y axis) and Nx columns (x axis)
    route_temp.T = 0;
    route_temp.latest_w = 0; % Initial moving direction
    route_temp.latest_v = 0.25; % Initial running speed
    route_temp.latest_x = r_start(1); % Initial x position
    route_temp.latest_y = r_start(2); % Initial y position

    Vs_temp = route_temp.latest_v;
    RDs_temp = route_temp.latest_w;
    positions_temp = r_start;
    
    % Generate a running trajectory starting from the left side of the place field
    for t = 0 : route_temp.dt : 100
        figure_on = 0;
        [v_temp, rd_temp, position_temp, ~, route_temp] = generate_virtual_rat_trajectory(route_temp.dt, env, rat_run_temp, route_temp, figure_on);
        %     toc
        if norm(position_temp-rc)>=R
            break;
        end
        positions_temp = [positions_temp; position_temp];
        Vs_temp = [Vs_temp v_temp];
        RDs_temp = [RDs_temp rd_temp];
        
        pause(0.01)
    end
    
    N_temp = length(RDs_temp);
    
    phis = [];
    PDCDs = [];
    colors = [];
    font_size = 12;
    
    for i_point = 1 : round(N_temp/num_points) : N_temp        
        r = positions_temp(i_point,:);
        RD = RDs_temp(i_point); % bottom to top
        
        % Compute the firing phase
        r_rc = complex(rc(1)-r(1),rc(2)-r(2));
        w = angle(r_rc) - deg2rad(RD);
        R_phi = sqrt(R^2 - abs(r_rc)^2*(sin(w))^2);
        delta_r = abs(r_rc)*cos(w);
        r_phi = R_phi - delta_r;
        
        
        % Initialisation for place cells
        U_place = zeros(lca.num_place_cell, Nt); % Membrane potentials of M neurons for batch_size images
        S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images
        [S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
            weak_network, grid_network, lca, r, RD, ts, env, S_place, U_place);
        
        
        fig = figure(1);
        fig.set( 'name', 'phase precession', 'units','centimeters', 'Position',[15 15 8 6]);
        fig_temp = plot(ts, S_place(i_place_cell,:)', 'linewidth', 2);
        colors = [colors; fig_temp.Color];
        hold on
        
        [params, fit_parameters] = fit_spatiotemporal(S_place(i_place_cell,:), ts);
        
        if ~isempty(params)
            PDCDs = [PDCDs -delta_r/R_phi];
            phis = [phis rad2deg(params(3))];
        end
    end
    
    if phis(end)>300
        phis(end) = phis(end) - 360;
    end
    
    fig = figure(2);
    fig.set( 'name', 'Phase vs. Position', 'units','centimeters', 'Position',[15 15 5 5]);
    scatter(PDCDs, phis, 20, colors, 'filled');
    xlim([-1 1])
    ylim([0 360])
    set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
    xlabel('pdcd')
    ylabel('firing phase (\circ)')
    set(gca,'FontSize',12)
    
    i_cell
    corr2(phis, PDCDs)
    
    PDCD_population = [PDCD_population PDCDs];
    phi_population = [phi_population phis];
    corrs(i_cell) = corr2(phis, PDCDs);
    entry_phis(i_cell) = phis(1);
    exit_phis(i_cell) = phis(end);

%     pause
end
close all

% Plot temporal property of the population using curved trajectories
font_size = 12;

%%%%%%%%%%%%%%%%%%%%%% Figure 4D in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure();
fig.set( 'name','4D(left)-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 5.5 4])
histogram(entry_phis, [-60 0 60 120 180 240 300 360 420]);
xlabel('entry phase (\circ)');
ylabel('counts');
ylim([0 65]);
set(gca, 'XTick', [0 180 360], 'XTickLabels', {0, 180, 360});
text(180,50,['N=' num2str(length(place_cells_index))],'fontsize',font_size)
set(gca,'FontSize',font_size);

fig = figure();
fig.set( 'name','4D(middle)-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 5.5 4])
histogram(exit_phis, [-60 0 60 120 180 240 300 360 420]);xlabel('exit phase (\circ)');
ylabel('counts');
ylim([0 65]);
set(gca, 'XTick', [0 180 360], 'XTickLabels', {0, 180, 360});
text(180,50,['N=' num2str(length(place_cells_index))],'fontsize',font_size)
set(gca,'FontSize',font_size);

fig = figure();
fig.set( 'name','4D(right)-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 5.5 4])
histogram(corrs, -1:0.2: 1);
xlabel('corr. coef.');
ylabel('counts');
% xlim([-1 -0.97]);
ylim([0 65]);
set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
text(0,50,['N=' num2str(length(place_cells_index))],'fontsize',font_size)
set(gca,'FontSize',font_size);

%%%%%%%%%%%%%%%%%%%%%% Figure 4C in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure();
fig.set( 'name','4C-Histogram of entry phase', 'units','centimeters', 'Position',[15 15 10 7])
scatter(PDCD_population(PDCD_population>-0.999), phi_population(PDCD_population>-0.999),2, 'filled')
xlabel('pdcd');
ylabel('firing phase (\circ)')
xlim([-1 1]);
ylim([0 360]);
axis square
set(gca, 'XTick', [-1 0 1], 'XTickLabels', {-1, 0, 1});
set(gca, 'YTick', [0 180 360], 'YTickLabels', {0, 180, 360});
set(gca,'FontSize',font_size);
% c = polyfit(PDCD_population(PDCD_population>-0.999),phi_population(PDCD_population>-0.999),1);


%% Figure 5 in the manuscript (Scenario 3)
% After inactivating MEC grid cells
close all
clear
clc
addpath('functions')
addpath('results')

% The result dataset used in the paper cannot be uploaded to Github due to its size.
% You are welcome to contact yanbo.lian@unimelb.edu.au to request the data
load('Scenario_2&3.mat')

font_size = 28;

% Select model place cells
fit_errors = [lca.fields_fit_no_grid(:).fit_error];
% max(fit_errors)

fit_error_threshold = 40; % In percentage
radius_threshold = 0.05; % Unit: meters.
width_scale = 0;

% Find place cells that meet the criteria: fitting error<40% and radius>5cm
place_cells_index_no_grid = [];
for i_cell = 1 : lca.num_place_cell
    if lca.fields_fit_no_grid(i_cell).fit_error < fit_error_threshold ...
            && lca.fields_fit_no_grid(i_cell).params(4) > radius_threshold ...
            && lca.fields_fit_no_grid(i_cell).params(2) > width_scale*lca.fields_fit_no_grid(i_cell).params(4)...
            && lca.fields_fit_no_grid(i_cell).params(2) < env.x_size-width_scale*lca.fields_fit_no_grid(i_cell).params(4) ...
            && lca.fields_fit_no_grid(i_cell).params(3) > width_scale*lca.fields_fit_no_grid(i_cell).params(4)...
            && lca.fields_fit_no_grid(i_cell).params(3) < env.y_size-width_scale*lca.fields_fit_no_grid(i_cell).params(4) ...
        place_cells_index_no_grid = [place_cells_index_no_grid i_cell];
    end
end

%%%%%%%%%%%%%%%%%%%%%% Figure 5A in the manuscript %%%%%%%%%%%%%%%%%%%%%
disp = lca.place_field_recovered_no_grid(:, place_cells_index_no_grid);
fig = figure(1);
fig.set( 'name',['5A (left)-' num2str(length(place_cells_index_no_grid)) ' place cells'],'units','normalized','position',[0.1 0.1 0.45 0.7]);
display_matrix(disp, 3);
colormap(jet_modified)
h = colorbar;
h.Limits = [0 1];
h.Ticks = [0 0.1 1];
h.TickLabels = {'0', '10%', 'Max'};
h.FontSize = 28;

% Plot centers of place cells
Nx = 80;
Ny = 80;
delta_x = env.x_size / (Nx - 1);
delta_y = env.y_size / (Ny - 1);
place_fields_centers = zeros(Nx, Ny);
for i_cell = place_cells_index_no_grid
    i_x = min(max(1 + round(lca.fields_fit_no_grid(i_cell).params(2)/delta_x), 1),Nx);
    i_y = min(max(1 + round(lca.fields_fit_no_grid(i_cell).params(3)/delta_y), 1),Ny);
    place_fields_centers(i_x, i_y) = 1;
end
figure(2)
imagesc(place_fields_centers);
axis equal
colormap(gray)
axis image
% set(gca, 'YDir', 'normal')
set(gca, 'XTick', [1,Nx/2,Nx], 'XTickLabels', {'0', [num2str(env.x_size/2) ' m'],[num2str(env.x_size) ' m']})
set(gca, 'YTick', [1,Ny/2,Ny], 'YTickLabels', {'0', [num2str(env.y_size/2) ' m'],[num2str(env.y_size) ' m']})
set(gca, 'FontSize', font_size);

%%%%%%%%%%%%%%%%%%%%%% Figure 5B in the manuscript %%%%%%%%%%%%%%%%%%%%%
fit_errors = [lca.fields_fit(:).fit_error];

fit_error_threshold = 40; % In percentage;
radius_threshold = 0.05; % Unit: meters.
width_scale = 0;

% Find place cells that meet the criteria: fitting error<15% and radius>5cm and within the environment
place_cells_index = [];
for i_cell = 1 : lca.num_place_cell
    if lca.fields_fit(i_cell).fit_error < fit_error_threshold ...
            && lca.fields_fit(i_cell).params(4) > radius_threshold ...
            && lca.fields_fit(i_cell).params(2) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(2) < env.x_size-width_scale*lca.fields_fit(i_cell).params(4) ...
            && lca.fields_fit(i_cell).params(3) > width_scale*lca.fields_fit(i_cell).params(4)...
            && lca.fields_fit(i_cell).params(3) < env.y_size-width_scale*lca.fields_fit(i_cell).params(4) ...
        place_cells_index = [place_cells_index i_cell];
    end
end

place_index_intersect = intersect(place_cells_index,place_cells_index_no_grid);

radiuses = zeros(length(place_index_intersect), 1);
for i_cell = 1 : length(place_index_intersect)
    radiuses(i_cell) = lca.fields_fit(place_index_intersect(i_cell)).params(4);
end
fprintf(['mean radius: ' num2str(mean(100*radiuses)) ', std radius: ' num2str(std(100*radiuses)) '\n']);

radiuses = zeros(length(place_index_intersect), 1);
for i_cell = 1 : length(place_index_intersect)
    radiuses(i_cell) = lca.fields_fit_no_grid(place_index_intersect(i_cell)).params(4);
end
fprintf(['(after inactivation) mean radius: ' num2str(mean(100*radiuses)) ', std radius: ' num2str(std(100*radiuses)) '\n']);


delta_radii = zeros(size(place_index_intersect));
delta_centers = zeros(size(length(place_index_intersect),2));
for i_cell = 1 : length(place_index_intersect)
    delta_radii(i_cell) = lca.fields_fit_no_grid(place_index_intersect(i_cell)).params(4) - ...
        lca.fields_fit(place_index_intersect(i_cell)).params(4);
    
    delta_centers(i_cell, 1) = lca.fields_fit_no_grid(place_index_intersect(i_cell)).params(2) - ...
        lca.fields_fit(place_index_intersect(i_cell)).params(2);
    delta_centers(i_cell, 2) = lca.fields_fit_no_grid(place_index_intersect(i_cell)).params(3) - ...
        lca.fields_fit(place_index_intersect(i_cell)).params(3);
end

%%%%%%%% Figure 5B in the manuscript
fig = figure(3);
font_size = 12;
fig.set( 'name','5B-Histogram of radius change', 'units','centimeters', 'Position',[15 15 8 6]);
histogram(100*delta_radii, -5:1:5);
xlabel('radius change (cm)');
ylabel('counts');
text(-2,25,['N=' num2str(length(place_index_intersect))],'fontsize',font_size)
set(gca, 'XTick', [-4 -2 0 2 4], 'XTickLabels', {-4,-2,0,2,4});
set(gca,'FontSize',font_size);

%%%%%%%%%%%%%%%%%%%%%% Figure 5C in the manuscript %%%%%%%%%%%%%%%%%%%%%
fig = figure(4);
fig.set( 'name','5C-Scatter plot of center change', 'units','centimeters', 'Position',[15 15 6 6]);
plot(100*delta_centers(:,1), 100*delta_centers(:,2), '.', 'markersize',9)
xlabel('centre shift in x (cm)');
ylabel('centre shift in y (cm)');
xlim([-13 13])
ylim([-13 13])
axis square
set(gca,'FontSize',font_size);
grid on;
text(4.5,7,['N=' num2str(length(place_index_intersect))],'fontsize',font_size)









