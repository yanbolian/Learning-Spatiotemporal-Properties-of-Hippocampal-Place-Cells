%% This use sparse coding (LCA) to learn place fields given entorhinal cells as the input
% It implements psrase coding with non-negative constraint

%% Create place cells (definition of symbols)
lca.A = normalize_matrix(rand(weak_network.num_cells+grid_network.num_cells, lca.num_place_cell)); % Connections between input and output layer
U_place = zeros(lca.num_place_cell, 1); % Membrane potentials of M neurons for batch_size images
S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images

%% Load training trajectory
load(trajectory.file_name);
trajectory.dt = route.dt;
trajectory.T = route.T;
trajectory.Nt = length(ts);
trajectory.dt = route.dt; % in second
lca.n_iter = floor(trajectory.dt / lca.dt);

Trajectory = zeros(env.Ny, env.Nx); % the matrix to store updated discretized positions

%% main loop
% The learning process when the virtual rat explroes the 2D environment

residual = 0;
x = 0;
s = 0;
A_eta = lca.A_eta;

for i = 1 : trajectory.Nt
    
    r = positions(i,:);
    RD = RDs(i);
    t = ts(i);
    
    % discretized trajectories; used for plotting
    Trajectory(round(r(2)/env.delta_y)+1, round(r(1)/env.delta_x)+1) = ...
        Trajectory(round(r(2)/env.delta_y)+1, round(r(1)/env.delta_x)+1) + 1;
    
    [S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
        weak_network, grid_network, lca, r, RD, t, env, S_place, U_place);
    
    % apply the learning rule
    R = S - lca.A * S_place; % Calculate residual error
    dA1 = R * S_place';
    lca.A = lca.A + A_eta * dA1;
    lca.A = max(lca.A, 0); % A is non-negative
    lca.A = normalize_matrix(lca.A, 'L2 norm', 1); % Normalize each column of the connection matrix
    
    % Save R_average, X_average and A_error
    R_average(i) = sum(sum(R.^2));
    X_average(i) = sum(sum(S.^2));
    
    residual = residual + R_average(i);
    x = x + X_average(i);
    s = s + sum(S_place(:)~=0) / lca.num_place_cell;
    
    % display 
    if (mod(i, lca.display_every/trajectory.dt) == 0)
        fprintf('%3.1f s: Percentage of active units: %3.1f%%, MSE: %3.1f%%\n',...
            i*trajectory.dt, 100 * sum(S_place(:) ~= 0) / lca.num_place_cell, 100 * R_average(i) / X_average(i)...
            );
        
        % Display connection matrix, A, and responses, S
        figure(1)
        subplot (331);display_matrix(grid_network.G_matrix, 3); title('Grid network-G'); colormap(gray);colorbar
        subplot (332);display_matrix(lca.A(weak_network.num_cells+1:end,:), 1); title('A'); colormap(gray);colorbar
        subplot (333);display_matrix(grid_network.G_matrix * lca.A(weak_network.num_cells+1:end,:), 3); title('G*A'); colormap(jet_modified);colorbar
        
        if weak_network.num_cells ~=0
            subplot (334);display_matrix(weak_network.G_matrix, 3); title('Weak network-W'); colormap(gray);colorbar
            subplot (335);display_matrix(lca.A(1:weak_network.num_cells,:), 1); title('A'); colormap(gray);colorbar
            subplot (336);display_matrix(weak_network.G_matrix * lca.A(1:weak_network.num_cells,:), 3); title('W*A'); colormap(jet_modified);colorbar
            
            subplot (337);display_matrix([weak_network.G_matrix grid_network.G_matrix], 3); title('Entorhinal-E'); colormap(gray);colorbar
            subplot (338);display_matrix(lca.A, 1); title('A'); colormap(gray);colorbar
            subplot (339);display_matrix([weak_network.G_matrix grid_network.G_matrix] * lca.A, 3); title('E*A'); colormap(jet_modified);colorbar
        end
        
        figure(2);
        subplot(311);stem(S);title('Input');
        subplot(312);stem(S_place);title('S: the firing rates');
        subplot(313);stem(U_place);title('U: the membrane potentials');
        
        if lca.history_flag == 1
            figure(3);
            plot(S_his);title('S: the firing rates');
        end
        
        figure(4);
        imagesc(Trajectory);
        axis image
        set(gca, 'YDir', 'normal')
        set(gca, 'XTick', [1,env.Nx/2,env.Nx], 'XTickLabels', [0,env.x_size/2,env.x_size])
        set(gca, 'YTick', [1,env.Ny/2,env.Ny], 'YTickLabels', [0,env.y_size/2,env.y_size])
        
        fprintf('Average Percentage of active units: %3.1f%%\n', 100 * s / (lca.display_every/trajectory.dt));
        fprintf('MSE: %3.1f%%\n', 100 * residual / x);
        
        residual = 0;
        x = 0;
        s = 0;
        
        pause(0.1)
    end
    
% pause
end

%% Recover the place fields using testing trajectory
load(trajectory.test_file_name);
trajectory.T_test = route.T;
trajectory.Nt_test = length(ts);
dt = route.dt; % in second

input_positions = discretize_positions(positions, env);

U_place = randn(lca.num_place_cell, 1); % Membrane potentials of M neurons for batch_size images
S_place = max(U_place - lca.lambda, 0); % Firing rates (Response) of M neurons for batch_size images

figure(5);
imagesc(route.Trajectory);
axis image
set(gca, 'YDir', 'normal')
set(gca, 'XTick', [1,env.Nx/2,env.Nx], 'XTickLabels', [0,env.x_size/2,env.x_size])
set(gca, 'YTick', [1,env.Ny/2,env.Ny], 'YTickLabels', [0,env.y_size/2,env.y_size])

S_test = [];
for i = 1 : trajectory.Nt_test
    r = positions(i,:);
    RD = RDs(i);
    t = ts(i);
    
    [S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
        weak_network, grid_network, lca, r, RD, t, env, S_place, U_place);
        
    S_test(:,i) = S_place;
    
     if (mod(i, lca.display_every/dt) == 0)
        fprintf('%3.1f s: Percentage of active units: %3.1f%%\n',...
            i*dt, 100 * sum(S_place(:) ~= 0) / lca.num_place_cell);
     end
end

epsilon = 1e-16;
lca.place_field_recovered = input_positions * S_test' ./ ( repmat(sum(S_test,2)', env.L, 1) + epsilon ); % Using reverse correlation / STA to recover the place field of cells
lca.ACTIVE_RATE = sum(S_test(:)~=0) / trajectory.Nt_test / lca.num_place_cell;

%% Recover the place fields when grid cells are inactive (only for Scenario 3 in the paper)
if weak_network.num_cells ~= 0
    
    lca_no_grid = lca;
    
    % Normalize the connections from weakly spatial population to place cells
    lca_no_grid.A(1:weak_network.num_cells,:) = normalize_matrix(lca_no_grid.A(1:weak_network.num_cells,:));
    lca_no_grid.A(weak_network.num_cells+1:end,:) = 0;
    
    U_place = randn(lca_no_grid.num_place_cell, 1); % Membrane potentials of M neurons for batch_size images
    S_place = max(U_place - lca_no_grid.lambda, 0); % Firing rates (Response) of M neurons for batch_size images
    
    S_test = [];
    for i = 1 : trajectory.Nt_test
        r = positions(i,:);
        RD = RDs(i);
        t = ts(i);
        
        [S, S_place, U_place, S_his, ~] = compute_place_cells_spatiotemporal_responses(...
            weak_network, grid_network, lca_no_grid, r, RD, t, env, S_place, U_place);
        
        S_test(:,i) = S_place;
        
        if (mod(i, lca_no_grid.display_every/dt) == 0)
            fprintf('%3.1f s: Percentage of active units: %3.1f%%\n',...
                i*dt, 100 * sum(S_place(:) ~= 0) / lca_no_grid.num_place_cell);
        end
    end
    
    epsilon = 1e-16;
    lca.place_field_recovered_no_grid = input_positions * S_test' ./ ( repmat(sum(S_test,2)', env.L, 1) + epsilon ); % Using reverse correlation / STA to recover the place field of cells
    lca.ACTIVE_RATE_no_grid = sum(S_test(:)~=0) / trajectory.Nt_test / lca.num_place_cell;
end



