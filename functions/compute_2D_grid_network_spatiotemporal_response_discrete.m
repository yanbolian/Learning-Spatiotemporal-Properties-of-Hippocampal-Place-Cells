function responses = compute_2D_grid_network_spatiotemporal_response_discrete(r, r_d, RD, ts, grid_network)
% r_d: N * 1 vector that respresents discrete position
% RD: current running direction, in degrees; right is 0 degree.
% ts: time stamps; temporal responses at location r
% grid_network: grid network that is a population of 2D grid cells
% responses: N * Nt vector

responses = zeros(grid_network.num_cells,length(ts));

%% Compute the responses of all grid cells in the grid network
for i_grid_cell = 1 : grid_network.num_cells    
    responses(i_grid_cell, : ) = compute_2D_grid_cell_spatiotemporal_response_discrete(r, r_d, RD, ts, grid_network.cells(i_grid_cell));
end