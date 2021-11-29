function nearest_wall = find_nearest_wall(x, y, x_size, y_size, bound_buffer)

nearest_wall = '';
if x>=0 && x<bound_buffer && y>=x && y<=(y_size-x)
    nearest_wall = 'left';
end

if x>x_size-bound_buffer && x<=x_size && y<=(x+y_size-x_size) && y>=(x_size-x)
    nearest_wall = 'right';
end

if y>=0 && y<bound_buffer && x>=y && x<=(x_size-y)
    nearest_wall = 'bottom';
end

if y>=y_size-bound_buffer && y<=y_size && x<=(y+x_size-y_size) && x>=(y_size-y)
    nearest_wall = 'top';
end