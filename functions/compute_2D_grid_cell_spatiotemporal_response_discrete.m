function responses = compute_2D_grid_cell_spatiotemporal_response_discrete(r, r_d, RD, ts, grid_cell_2D)
% r: [x, y] current position, in m; 
% RD: current running direction, in degrees; right is 0 degree.
% ts: time stamps; temporal responses at location r
% grid_network: grid network that is a population of 2D grid cells
% responses: N * Nt vector

f_theta = 10; % Hz; theta frequency
responses = zeros(1, length(ts));

%% Compute the responses of the 2D grid cells
vertices = grid_cell_2D.vertices;
radii = grid_cell_2D.radii;
amplitudes = grid_cell_2D.amplitudes;
amplitude_scale = grid_cell_2D.amplitude_scale;

k_phi = grid_cell_2D.k_phi;
phi_0 = grid_cell_2D.phi_0;
delta_phi = grid_cell_2D.delta_phi;


% find the nearest vertices relative to the current position
[~, ind] = min(sum((r-vertices).^2,2));
rc = vertices(ind,:);
R = radii(ind);
amplitude = amplitudes(ind);

% Compute the firing phase
r_rc = complex(rc(1)-r(1),rc(2)-r(2));

if abs(r_rc) <= R
    w = angle(r_rc) - deg2rad(RD);
    R_phi = sqrt(R^2 - abs(r_rc)^2*(sin(w))^2);
    delta_r = abs(r_rc)*cos(w);
    r_phi = R_phi - delta_r;
    firing_phase = phi_0 - delta_phi * r_phi / 2 / R_phi;
%     mod(firing_phase, 360)
    
    spatial_response = grid_cell_2D.G' * r_d;
    temporal_response = exp(k_phi * (cos(2*pi*f_theta*ts - deg2rad(firing_phase)) - 1) );
    responses(:) = amplitude_scale * spatial_response * temporal_response;

else
    % No response with distance > radius of the field
    responses(:) = 0;
end



end