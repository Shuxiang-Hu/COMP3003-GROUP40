clear;
clc;

%% Load data and Related element
% pre-set element
NUM_LABEL0_DATA = 100;
NUM_LABEL1_DATA = 100;
K_FOLD = 10;

% kernel function
TASK1_KF = "linear";
TASK2_KF = "rbf";

% load data
label0_data = importdata("datalabel0.txt");
label1_data = importdata("datalabel1.txt");
NUM_LABEL0_DATA = min(NUM_LABEL0_DATA, size(label0_data, 1));
NUM_LABEL1_DATA = min(NUM_LABEL1_DATA, size(label1_data, 1));

% Get the dataset and shuffle the list
data_all = [label0_data(1:NUM_LABEL0_DATA,:); label1_data(1:NUM_LABEL1_DATA,:)];
ir = randperm(NUM_LABEL0_DATA + NUM_LABEL1_DATA);
data_all = data_all(ir,:);

%% Task1 Linear model
task1_start = tic;
model_classification = fitcsvm(data_all(:,1:5),data_all(:,6), 'KernelFunction',TASK1_KF, 'BoxConstraint',1);
task1_elapsed = toc(task1_start);

fprintf("Linear SVM training done in: %f seconds.\n",task1_elapsed);

%% Task2 - Brute Force
% parameter range setting
param_range.c = 10.^(-3:3);
% kernel methods
if TASK2_KF == "rbf"
    param_range.sigma = 10.^(-3:3);
    [c, sigma] = ndgrid(param_range.c, param_range.sigma);
    param_grid = [sigma(:) c(:)];
elseif TASK2_KF == "polynomial"
    param_range.q = (2:4);
    [c, q] = ndgrid(param_range.c, param_range.q);
    param_grid = [q(:) c(:)];
else
    error("Invalid kernel function");
end

% parameter optimisation
task2_start = tic;
[op_stats, optimise_hyperparameters, opt_acc] = innerCV(data_all(:, 1:5), data_all(:,6), TASK2_KF, param_grid, K_FOLD);
tast2_elapsed = toc(task2_start);

% result
fprintf("Optimisation of hyper-parameter done in: %f seconds.\n",tast2_elapsed);
% for i = 1 : length(op_stats)
%     fprintf("Combination: %d | c: %f | sigma: %f\n", i, op_stats(i).c, op_stats(i).sigma);
% end
fprintf("\nOptimise Combition is sigma/q: %f, c: %f\n", optimise_hyperparameters(1,1), optimise_hyperparameters(1,2));
