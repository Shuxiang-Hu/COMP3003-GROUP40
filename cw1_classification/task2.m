clear;
clc;

%% Load data and Related element
% pre-set element


NUM_LABEL0_DATA = 1900;
NUM_LABEL1_DATA = 1900;

K_FOLD = 10;

% kernel function
TASK2_KF = "polynomial";

% load data
label0_data = importdata("datalabel0.txt");
label1_data = importdata("datalabel1.txt");
NUM_LABEL0_DATA = min(NUM_LABEL0_DATA, size(label0_data, 1));
NUM_LABEL1_DATA = min(NUM_LABEL1_DATA, size(label1_data, 1));

% Get the dataset and shuffle the list
data_all = [label0_data(1:NUM_LABEL0_DATA,:); label1_data(1:NUM_LABEL1_DATA,:)];
ir = randperm(NUM_LABEL0_DATA + NUM_LABEL1_DATA);
data_all = data_all(ir,:);

%% Task2 - Brute Force with Kfold (Task3)
% parameter range setting

param_range.c = 2.^(-3:3);
kfold = 10;
lfold = 10;
x = data_all(:, 1:4);
y = data_all(:,5);



if TASK2_KF == "rbf"
    param_range.sigma = 2.^(-3:3);
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

train(x,y,kfold,lfold,TASK2_KF,param_grid);

tast2_end = toc(task2_start);
fprintf("Optimisation of hyper-parameter done in: %f seconds.\n",tast2_end);