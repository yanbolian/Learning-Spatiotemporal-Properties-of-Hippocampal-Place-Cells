function G = generate_2D_weakly_modulated(env, sigma_analog)
% G: returned 2D grid field normalised to [0, 1] with size Ny * Nx (Ny rows and Nx columns)
% env: paramters that define the spatial environment
% sigma_analog: unit m; smoothing Gaussian kernel in Neher et al. 2017, 0.06

G = rand(env.Ny, env.Nx);
sigma_discrete = sigma_analog / env.x_size * env.Nx; % Assume x_size:y_size = Nx:Ny

G = imgaussfilt(G, sigma_discrete);

G = (G - min(G(:))) / (max(G(:)) - min(G(:))) ; % normalize G to [0 1]