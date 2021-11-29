%% Fit place fields to 2D Gaussian function

[lca.fields_fit, lca.fit_parameters] = fit_2D_gaussian(lca.place_field_recovered, env.x_size, env.y_size, env.Nx, env.Ny);

if weak_network.num_cells ~= 0
    [lca.fields_fit_no_grid, lca.fit_parameters_no_grid] = fit_2D_gaussian(lca.place_field_recovered_no_grid, env.x_size, env.y_size, env.Nx, env.Ny);
end