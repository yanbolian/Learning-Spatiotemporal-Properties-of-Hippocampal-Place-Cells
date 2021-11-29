function [vertices, radii, amplitudes, G] = generate_2D_grid_fields(...
     lambda, orien, phase, amplitude_std, amplitude_scale, env)
%% This function generate grid fields by returning vertices, radiuses and amplitudes of each field
% G: returned 2D grid field with size Ny * Nx (Ny rows and Nx columns)
% vertices: returned vertice of the grid field
% radiuses: returned radius of each field
% amplitudes: returned amplitude of each field
% x_size, y_size: size of the environment (unit: m)
% Nx, Ny: number of discrete points in the x and y axis
% lambda: grid spacing (unit: m)
% orien: orientation of the grid pattern (unit: degrees); corresponding to the vertical line in a clockwise manner
% phase: grid phase - (x0,y0) (unit: m); 0~sqrt(3)*lambda
% From Neher et al. 2017
% Author: Yanbo Lian
% Date 20201020

if ~exist('env','var')
    env.x_size = 1; % unit: m
    env.y_size = 1; % unit: m
    env.bin_size = 0.02; % unit: m; discretize the continuous env. to discrete bins
    env.Nx = round(env.x_size/env.bin_size); % number of discrete points in x-axis
    env.Ny = round(env.y_size/env.bin_size); % number of discrete points in y-axis
end

orien = deg2rad(orien);
%% The statistics of amplitudes and radius of each field
% There are noise on the location of the vertex and amplitude for each vertex
amplitude_mean = 1; % Mean of amplitude
% amplitude_std = 0; % Standard deviation of amplitude for each location

radius = 0.32 * lambda; % The radius of 2D Gaussian function of each field
x0 = phase(1); % Phase offset in x-axis
y0 = phase(2); % Phase offset in y-axis

% Suppose the hexagonal pattern starts from the origion
% centers of hexagonal vertice can be represented by
% [kx*lambda, ky*sqrt(3)*lambda], [(kx+0.5)*lambda, (ky+0.5)*sqrt(3)*lambda]
kx_max = 2 * ceil((env.x_size+2*radius)/lambda);
ky_max = 2 * ceil((env.y_size+2*radius)/lambda);
k_min = -max(kx_max, ky_max);

%% Determine the parameters of each grid field
vertices = [];
radii = [];
amplitudes = [];
for kx = k_min : 1 : kx_max
    for ky = k_min : 1 : ky_max
        amplitude = amplitude_mean + amplitude_std * randn;
        vertice = [kx*lambda*cos(orien)+ky*sqrt(3)*lambda*sin(orien)+x0, ...
                    -kx*lambda*sin(orien)+ky*sqrt(3)*lambda*cos(orien)+y0];
        
        % Store vertice, radius and amplitude of the grid cell
        if is_in_bound(vertice(1), vertice(2), env.x_size, env.y_size, 2*radius)
            vertices = [vertices; vertice];
            radii = [radii radius];
            amplitudes = [amplitudes amplitude];
        end
        
        amplitude = amplitude_mean + amplitude_std * randn;
        vertice = [(kx+0.5)*lambda*cos(orien)+(ky+0.5)*sqrt(3)*lambda*sin(orien)+x0, ...
                    -(kx+0.5)*lambda*sin(orien)+(ky+0.5)*sqrt(3)*lambda*cos(orien)+y0];
        
        % Store vertice, radius and amplitude of the grid cell
        if is_in_bound(vertice(1), vertice(2), env.x_size, env.y_size, 2*radius)
            vertices = [vertices; vertice];
            radii = [radii radius];
            amplitudes = [amplitudes amplitude];
        end
    end
end

%%
% Generate XY coordinates in the extended environment
[X, Y] = meshgrid(linspace(0,env.x_size,env.Nx), linspace(0, env.y_size, env.Ny));
% X Y is in accordance with normal xy-axis
% Y: top to bottom ~ 0 to 1m

G = zeros(size(X));

fun_2D_gaussian = @(vertice, radius, amplitude) ...
    amplitude * exp( -log(5) * ((X-vertice(1)).^2+(Y-vertice(2)).^2) / radius^2 ); % Eq 1 in Neher et al. 2017

num_vertices = length(amplitudes);

for i_vertice = 1 : num_vertices
    vertice = vertices(i_vertice,:);
    radius = radii(i_vertice);
    amplitude = amplitudes(i_vertice);
    G = G + fun_2D_gaussian(vertice, radius, amplitude);
end

G = amplitude_scale * G;

%% The function to determine whether vertices are within the region considered
function in_bound = is_in_bound(x, y, x_bound, y_bound, buffer)
if (x <= x_bound+buffer) && (y <= y_bound+buffer) && (x >= 0-buffer) && (y >= 0-buffer)
    in_bound = 1;
else
    in_bound = 0;
end
end

end