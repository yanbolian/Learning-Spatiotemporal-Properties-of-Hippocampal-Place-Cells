function [S,U,S_history, U_history] = sparse_coding_by_LCA(...
        X, A, lambda, thresh_type, eta, n_iter, history_flag, S_past, U_past)
% Implement sparse coding using local competition algorithm, i.e. LCA (Rozzel et al. 2008)
% X: input matrix where each column is one example of the input
% A: the columns of A are basis vectors; the number of columns are the
% number of input neurons
% lambda: threshold of thresholding function
% tau: time constant
% n_iter: maximum number of iteration
% thresh_type: thresh_type of thresholding function
% history_flag: record the trajectories or not
% S_past, U_past: previous firing rates and membrane potentials

s_max = 10; % Maximum of neuronal response

S_history = []; % Dynamics of the firing rates; each row is the responses of all neurons for the current iteration
U_history = []; % Dynamics of the membrane potentials

W = A' * A - eye(size(A, 2));
U_init = A' * X; % Initial membrane potentials of the neuron

if exist('S_past','var') && exist('U_past','var')
    S = S_past;
    U = U_past;
else
    U = zeros(size(U_init));
    S = zeros(size(U_init));
end

% The process of computing neuronal responses
for i = 1 : n_iter
    % Save the history (trajectory) of model dynamics
    if exist('history_flag','var') && (history_flag == 1)
        S_history(i,:) = S(:,1); % Record the history of the first input
        U_history(i,:) = U(:,1); % Record the history of the first input
    end

    % Compute the membrane potential, U, of the neuron
    delta_U = eta * (U_init - U - W*S);
    U = U + delta_U;
    
    % Get firing rate, S, of the neuron by thresholding the membrane, U
    if isequal(thresh_type, 'soft')
        S = wthresh(U, 's', lambda);
    elseif isequal(thresh_type, 'hard')
        S = wthresh(U, 'h', lambda);
    elseif isequal(thresh_type,'hard-non-negative')
        S = max(wthresh(U, 'h', lambda), 0);
    elseif isequal(thresh_type, 'soft-non-negative')
        S = max(U - lambda, 0);
    elseif isequal(thresh_type, 'sigmoid')
        alpha = 0;% [0,1]
        gamma = 5;
        lambda = 1;
        S = (U - alpha*lambda) ./ (1 + exp(-gamma * (U-lambda)));
    else
        error('Not a suitable threshold type. Check thresh_type');
    end
    S = min(S, s_max);
end



