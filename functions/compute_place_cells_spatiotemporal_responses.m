function [S, S_place, U_place, S_his, U_his] = compute_place_cells_spatiotemporal_responses(...
        weak_network, grid_network, lca, r, RD, ts, env, S_place_past, U_place_past)

    
% transform continuous r to r_discrete
r_d = discretize_positions(r, env);

if weak_network.num_cells == 0
    S_weak = [];
else
    S_weak = weak_network.G_matrix' * r_d;
end

% compute spatiotemporal responses of grid cells
if grid_network.discretization == 1
    S_grid = compute_2D_grid_network_spatiotemporal_response_discrete(r, r_d, RD, ts, grid_network);
else
    S_grid = compute_2D_grid_network_spatiotemporal_response(r, RD, ts, grid_network);
end

S_weak = repmat(S_weak, 1, length(ts));

S = [S_weak; S_grid];

if exist('S_place_past','var') && exist('U_place_past','var')
    [S_place, U_place, S_his, U_his] = sparse_coding_by_LCA(...
        S, lca.A, lca.lambda, lca.thresh_type, lca.U_eta, lca.n_iter, lca.history_flag, S_place_past, U_place_past);
else
    [S_place, U_place, S_his, U_his] = sparse_coding_by_LCA(...
        S, lca.A, lca.lambda, lca.thresh_type, lca.U_eta, lca.n_iter, lca.history_flag);
end