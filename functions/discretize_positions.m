function r_d = discretize_positions(r, env)
% 
% r: Nt*2
% env: parameters of the environment

delta_x = env.x_size / (env.Nx-1);
delta_y = env.y_size / (env.Ny-1);

N = size(r, 1);
r_d = zeros(env.Nx*env.Ny, N);

for i = 1 : N
    r_temp = zeros(env.Ny, env.Nx);
    r_temp(round(r(i,2)/delta_y)+1, round(r(i,1)/delta_x)+1) = 1;
    r_d(:,i) = reshape(r_temp, numel(r_temp), 1);
end