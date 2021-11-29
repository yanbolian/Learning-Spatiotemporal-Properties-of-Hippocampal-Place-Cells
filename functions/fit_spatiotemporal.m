%% Fit spatio-temporal response of a place cell to a function
function [params, fit_parameters] = fit_spatiotemporal(spatiotemporal_response, ts)

opts.plot_figure = 0; % if set to 1, then for each filter the original and fitted data will be plotted
opts.num_trials = 20; % number of trials in optimization loop
opts.fit_error_threshold = 0; % normalized fitting error threshold in percentage
opts.datetime = datetime; % Date and time of the fitting process

% params(1) = k_phi
% params(2) = F
% params(3) = phi
fun_spatiotemporal_response = @(params, ts) ...
    1 * exp(params(1)*(cos(2*pi*params(2)*ts-params(3))-1));

params = [];
fit_error = 100;

amplitude = max(spatiotemporal_response);
if amplitude <= 0
    params = [];
    fit_parameters = [];
else
    spatiotemporal_response = spatiotemporal_response / amplitude; % normalize the response
    for i_trials = 1 : opts.num_trials
        fprintf('(%d)', i_trials);
        params_initial = [1 10 0 0.8] + 0.1 * randn(1,4);
        options = optimset('Display', 'off', 'MaxIter', 1000);
        
        [params_current, resnorm] = lsqcurvefit(fun_spatiotemporal_response, ...
            params_initial, ts, spatiotemporal_response, [], [], options);
        
        epsilon = 1e-12; % avoid zero division
        fit_error_current = resnorm / (epsilon+sum(spatiotemporal_response.^2)) * 100; % compute the fit error percentage
        
        params_current(3) = params_current(3) + pi*(sign(params_current(1))==-1);
        params_current(1) = abs(params_current(1));
        
        % update fitError and determine whether to exit fitting process
        if fit_error_current < fit_error
            fit_error = fit_error_current;
            params = params_current;
        end
        if fit_error < opts.fit_error_threshold
            break;
        end
        
        % print
        if ~mod(i_trials, 10)
            fprintf(' min fit error reached = %.2f percent.\n', fit_error);
        end
    end
    fprintf('Done!\n');

    if ~isempty(params)
        params(3) = mod(params(3), 2*pi);
        params(4) = amplitude;
        spatiotemporal_response_fit = params(4) * fun_spatiotemporal_response(params(1:3), ts);
        
        if opts.plot_figure
            plot(ts, spatiotemporal_response_fit);
            xlabel('time (s)'); ylabel('response')
            title('Response over 1s: from the left to right of a place field')
            ylim([0,5]);
        end
    end
    
    params';
    fprintf('The normalized fitting error is %.2f percent.\n', fit_error);
    fit_parameters = opts;
end