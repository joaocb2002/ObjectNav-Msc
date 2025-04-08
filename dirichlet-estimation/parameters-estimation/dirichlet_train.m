function dirichlet_train(dirName, id, name)
% DIRICHLET_TRAIN Fits Dirichlet distributions to score vectors and saves the result.
%
% This function loads a matrix of softmax-like score vectors from a .mat file,
% groups them by category and bounding box range, and fits a Dirichlet distribution
% to each group using maximum likelihood estimation (via `dirichlet_fit`).
%
% INPUTS:
%   dirName - Output filename (with path) to save the struct of Dirichlet parameters
%   id      - Integer category ID (used for metadata)
%   name    - Category name (used for metadata)
%
% OUTPUT:
%   Saves a struct `s` with fields:
%     - s.a{i,j}   : Dirichlet parameters (alpha vector) for bin (i,j)
%     - s.id       : Category ID
%     - s.name     : Category name
%
% REQUIREMENTS:
%   - Expects a file '../tmp/scores.mat' with variable `Scores`, a cell array
%     where each cell contains a matrix of score vectors for a given (scale, position).
%   - Depends on the function `dirichlet_fit(data)`, which must be on the MATLAB path.

%% Load data
if ~exist('../tmp/scores.mat', 'file')
    error('Could not find ../tmp/scores.mat. Make sure the file is generated before calling this function.');
end
load('../tmp/scores.mat');  % Loads variable `Scores`

if ~exist('Scores', 'var')
    error('Variable ''Scores'' not found in scores.mat');
end

s.a = {};
s.id = id;
s.name = name;

%% Fit Dirichlet parameters for each bin
for i = 1:size(Scores,1)
    for j = 1:size(Scores,2)
        if isempty(Scores{i,j})
            continue;
        end

        % Normalize so each row sums to 1
        sums = sum(Scores{i,j},2);
        data = Scores{i,j} ./ max(sums, eps);

        % Fit only if we have enough samples
        if size(data,1) > 2
            [s.a{i,j}, ~] = dirichlet_fit(data);
        end
    end
end

%% Save result
save(dirName, 's');

end
