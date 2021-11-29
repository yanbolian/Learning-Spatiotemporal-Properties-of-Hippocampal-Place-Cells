function [new_w] = turn_near_wall(x, y, x_size, y_size, nearest_wall, w, bound_buffer)

switch nearest_wall
    case ''
        new_w = w;
    case 'left'
        if w>=pi && w<=1.5*pi
            if y >= bound_buffer
                new_w = 1.5 * pi;
            else
                new_w = 0;
            end
        elseif w>=0.5*pi && w<pi
            if y <= y_size - bound_buffer
                new_w = 0.5*pi;
            else
                new_w = 0;
            end
        else
            new_w = w;
        end
    case 'right'
        if w>=0 && w<=0.5*pi
            if y <= y_size - bound_buffer
                new_w = 0.5*pi;
            else
                new_w = pi;
            end
        elseif w>=1.5*pi && w<2*pi
            if y >= bound_buffer
                new_w = 1.5*pi;
            else
                new_w = pi;
            end
        else
            new_w = w;
        end
    case 'bottom'
        if w>1.5*pi && w<2*pi
            if x <= x_size - bound_buffer
                new_w = 0;
            else
                new_w = pi;
            end
        elseif w>=pi && w <=1.5*pi
            if x >= bound_buffer
                new_w = pi;
            else
                new_w = 0.5*pi;
            end
        else
            new_w = w;
        end
    case 'top'
        if w>=0 && w<0.5*pi
            if x <= x_size - bound_buffer
                new_w = 0;
            else
                new_w = 1.5*pi;
            end
        elseif w>=0.5*pi && w<=pi
            if x >= bound_buffer
                new_w = pi;
            else
                new_w = 1.5*pi;
            end
        else
            new_w = w;
        end
end