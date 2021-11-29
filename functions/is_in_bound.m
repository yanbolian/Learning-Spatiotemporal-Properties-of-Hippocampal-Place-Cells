function in_bound = is_in_bound(x, y, x_size, y_size)

if (x <= x_size) && (y <= y_size) && (x >= 0) && (y >= 0)
    in_bound = 1;
else
    in_bound = 0;
end