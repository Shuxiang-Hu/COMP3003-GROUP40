clear;
clc;

%% Load data and Related element
% pre-set element
NUM_LABEL0_DATA = 1000;
NUM_LABEL1_DATA = 1000;
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

%% Classification model
ctrain_start = tic;
model_classification = fitcsvm(data_all(:,1:5),data_all(:,6), 'KernelFunction',TASK1_KF, 'BoxConstraint',1);
ctrain_elapsed = toc(ctrain_start);
sv = model_classification.SupportVectors;

figure
gscatter(data_all(:,1),data_all(:,2),data_all(:,6))
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('occu','not','Support Vector')
hold off

%% task2 - Brute Force
param_range.c = 10.^(-3:3);

if TASK2_KF == "rbf"
    param_range.sigma = 10.^(-3:3);
elseif TASK2_KF == "polynomial"
    param_range.q = (2:4);
else
    error("Invalid kernel function");
end

if strcmp(TASK2_KF, "rbf") 
    [c, sigma] = ndgrid(param_range.c, param_range.sigma);
    param_grid = [sigma(:) c(:)];
elseif strcmp(TASK2_KF, "polynomial") 
    [c, q] = ndgrid(param_range.c, param_range.q);
    param_grid = [q(:) c(:)];
end

task2_start = tic;
[op_stats, optimise_hyperparameters, opt_acc] = innerCV(data_all(:, 1:5), data_all(:,6), TASK2_KF, param_grid, K_FOLD);
tast2_end = toc(task2_start);

fprintf("Optimisation of hyper-parameter done in: %f seconds.\n",tast2_end);
for i = 1 : length(op_stats)
    fprintf("Combination: %d | c: %f | sigma: %f\n", i, op_stats(i).c, op_stats(i).sigma);
end
fprintf("\nOptimise Combition is sigma/q: %f, c: %f\n", optimise_hyperparameters(1,1), optimise_hyperparameters(1,2));