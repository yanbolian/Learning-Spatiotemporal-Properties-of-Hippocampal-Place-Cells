%% 
% Author: Yanbo Lian
% Date: 202103-29

clear
close all
clc

addpath('functions')
addpath('results')
addpath('datasets')

%% Trajectory option
trajectory.option = 'virtual rat';
trajectory.file_name = 'rat_run_3600s.mat';
trajectory.test_file_name = 'rat_run_1200s.mat';

%% Creating the environment
env.x_size = 1; % unit: m
env.y_size = 1; % unit: m
env.bin_size = 0.025; % unit: m; discretize the continuous env. to discrete bins;
env.Nx = round(env.x_size/env.bin_size); % number of discrete points in x-axis
env.Ny = round(env.y_size/env.bin_size); % number of discrete points in y-axis
env.L = env.Nx * env.Ny;
env.delta_x = env.x_size / (env.Nx-1);
env.delta_y = env.y_size / (env.Ny-1);

%% Create a grid network with spatial and temporal properties
% Spatial properties: grid spacings, orientations and phases (similar to Lian & Burkitt 2021 eNeuro paper)
% Temporal properties: some parameters determines the phase precession of individual grid field

random_seed = 3;
rng(random_seed);

grid_network.num_cells = 900; 
grid_network.discretization = 1; % use discretized environment to compute grid cell responses
grid_network.amplitude_scale = 1; % maximum firing rate
grid_network.amplitude_std = 0.1; % amplitude of the white noise added to the maximum firing rate for individual grdi fields
grid_network.env = env;

% Grid spacing
grid_network.lambdas = [0.388 0.484 0.65 0.984]; % Mean grid spacings (unit: m) of the four discrete grid modules
grid_network.lambda_std = 0.08; % Standard deviation (unit: m) of grid spacings for each grid cell
grid_network.lambdas_perc = [0.87/2 0.87/2 0.13/2 0.13/2]; % Percentage of the grid population in the four discrete grid modules

% The number of grid cells in the four modules
grid_network.num_cells_module = zeros(1, length(grid_network.lambdas));
grid_network.num_cells_module(1:end-1) = floor(grid_network.lambdas_perc(1:end-1) * grid_network.num_cells);
grid_network.num_cells_module(end) = grid_network.num_cells - sum(grid_network.num_cells_module);

% Grid orientations
grid_network.orientations = [15 30 45 60]; % Mean orientations (unit: degree) of the four discrete grid modules
grid_network.orientation_std = 3; % Standard deviation (unit: degree) of grid orientations for each grid cell

% generate spacings and orientations for the grid population
lambdas = [];
orientations = [];
phases = [];
for i_module = 1 : length(grid_network.lambdas)
    lambdas =  ...
            [lambdas; ...
            grid_network.lambda_std*randn(grid_network.num_cells_module(i_module),1) + ...
            grid_network.lambdas(i_module) * ones(grid_network.num_cells_module(i_module),1)];
        
    orientations =  ...
            [orientations; ...
            grid_network.orientation_std*randn(grid_network.num_cells_module(i_module),1) + ...
            grid_network.orientations(i_module)*ones(grid_network.num_cells_module(i_module),1)];
end

% generate phases for the grid population according to the spacings by a uniforma distribution
for i_grid_cell = 1 : grid_network.num_cells
    phases(i_grid_cell, :) = [rand*lambdas(i_grid_cell) rand*lambdas(i_grid_cell)];
end

% generate each grid cell in the grid population
for i_grid_cell = 1 : grid_network.num_cells
    clear grid_cell_2D;
    
    % spatial properties of each grid cell
    grid_cell_2D.spacing = lambdas(i_grid_cell);
    grid_cell_2D.orientation = orientations(i_grid_cell); % in degrees
    grid_cell_2D.phase = phases(i_grid_cell, :);
    grid_cell_2D.amplitude_scale = grid_network.amplitude_scale;
    grid_cell_2D.amplitude_std = grid_network.amplitude_std;    
    
    % Temporal properties of each grid cell
    grid_cell_2D.k_phi = 0.8 + 0.4 * rand; % 
    grid_cell_2D.phi_0 = 300 + 40 * rand; % in degrees; entry phase
    grid_cell_2D.delta_phi = 300 + 40 * rand; % phase changes;
    
    % generate vertices, radii, and amplitudes for individual grid fields of each grid cell
    [grid_cell_2D.vertices, grid_cell_2D.radii, grid_cell_2D.amplitudes, G] = generate_2D_grid_fields(...
        grid_cell_2D.spacing, grid_cell_2D.orientation, grid_cell_2D.phase, grid_cell_2D.amplitude_std, grid_cell_2D.amplitude_scale, grid_network.env);
        
    % use a vector to represent the spatial field of a grid cell
    grid_cell_2D.G = reshape(G, numel(G), 1);
    grid_network.G_matrix(:, i_grid_cell) = grid_cell_2D.G;
    grid_network.cells(i_grid_cell) = grid_cell_2D;
end

% Display grid fields of the population
figure;
display_matrix(grid_network.G_matrix); colormap(jet_modified);
h = colorbar;
h.FontSize = 12;
h.Limits = [0 1];
h.Ticks = [0 0.1 1];
h.TickLabels = {'0', '10%', 'Max'};

r = [0.5 0.5];
RD = 0;
ts = 0: 0.01 : 2;
responses = compute_2D_grid_network_spatiotemporal_response(r, RD, ts, grid_network);
figure;
plot(ts, responses(1:4,:)');

%% MEC weakly tuned cells
clear weak_network

weak_network.num_cells = 400;
weak_network.sigma = 0.06; % unit: m; the smoothing Gaussian kernel in Neher et al. 2017
weak_network.amplitude_scale = 0.1; % the amplitude that scales the responses of weakly spatial cells
weak_network.env = env;


for i_cell = 1 : weak_network.num_cells
    G = weak_network.amplitude_scale * generate_2D_weakly_modulated(env, weak_network.sigma);
    weak_network.G_matrix(:, i_cell) = reshape(G, numel(G), 1);
end

if weak_network.num_cells ~= 0
    figure;
    display_matrix(weak_network.G_matrix); colormap(jet_modified);
    h = colorbar;
    h.FontSize = 12;
    h.Limits = [0 1];
    h.Ticks = [0 1];
    h.Limits = [0 1];
    h.Ticks = [0 0.1 1];
    h.TickLabels = {'0', '10%', 'Max'};
end

%% LCA: the parameters of local competitive algorithm (LCA) that implements sparse coding

lca.n_epoch = []; % Number of epoches
lca.num_place_cell = 100; % Number of neurons in the network
lca.lambda = 0.3;
lca.A_eta = 1e-2; % learning rate of A
lca.thresh_type = 'soft-non-negative';
lca.history_flag = 1;
lca.tau = 1e-2; % s; 10ms
lca.dt = 2e-4; % s; 0.2ms
lca.U_eta = lca.dt / lca.tau; % eta = dt / tau
lca.display_every = 10; % in seconds; Frequency of generating plot

%% Run the learning 
close all;

fprintf('%d weakly tuned spatial cells & %d grid cells in the network.\n', weak_network.num_cells, grid_network.num_cells);

entorhinal_to_place_spatiotemporal_lca;
fit_place_fields;
    
    
    