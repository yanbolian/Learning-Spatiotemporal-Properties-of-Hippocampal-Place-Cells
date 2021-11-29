function [] = display_2D_grid_cells(grid_cell_2D, env, figure_on)

% Display the vertices and 2D spatial grid field
if ~exist('figure_on','var')
    figure_on = 1;
end

if figure_on == 1
    buff = 3 * max(grid_cell_2D.radii);
    Nx_disp = 100;
    Ny_disp = round(Nx_disp * (env.y_size+2*buff) / (env.x_size+2*buff));
    
    % Plot vertices of grid fields
    
    delta_x = (env.x_size+2*buff) / (Nx_disp - 1); % [-x_size, 2*x_size]
    delta_y = (env.y_size+2*buff) / (Ny_disp - 1); % [-y_size, 2*y_size]
    grid_field_vertices = zeros(Nx_disp, Ny_disp);
    for i_cell = 1 : length(grid_cell_2D.amplitudes)
        i_x = 1 + round((grid_cell_2D.vertices(i_cell,1)+buff)/delta_x);
        i_y = 1 + round((grid_cell_2D.vertices(i_cell,2)+buff)/delta_y);
        grid_field_vertices(i_y, i_x) = 1; % y: top-bottom -> 0-1m
    end
    
    
    figure(1)
    subplot 121;
    imagesc(grid_field_vertices);
    title('vertices of the grid feild within 2*radius range of the boundary')
    % axis equal
    colormap(gray)
    axis image
    set(gca, 'YDir', 'normal')
    x0 = 1 + round(buff/delta_x);
    y0 = 1 + round(buff/delta_y);
    xm = 1 + round((env.x_size+buff)/delta_x);
    ym = 1 + round((env.y_size+buff)/delta_y);
    set(gca, 'XTick', [x0,xm], 'XTickLabels', {'0',[num2str(env.x_size) ' m']});
    set(gca, 'YTick', [y0,ym], 'YTickLabels', {'0',[num2str(env.y_size) ' m']});
    set(gca, 'FontSize', 12);
    
    [X, Y] = meshgrid(linspace(0,env.x_size,env.Nx), linspace(0, env.y_size, env.Ny));

    subplot 122;
    surf(X, Y, grid_cell_2D.G/max(grid_cell_2D.G(:)));
    shading interp
    set(gca,'YDir','reverse')
    view(0,90)
    colormap jet
    %     axis off
    axis image
    colorbar
end


