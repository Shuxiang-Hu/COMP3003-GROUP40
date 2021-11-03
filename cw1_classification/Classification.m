clear;
clc;

%% Load data and Related element
% pre-set element
NUM_LABEL0_DATA = 502;
NUM_LABEL1_DATA = 502;
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

% 10-fold cross-validation
crossval_start = tic;
model_cross = crossval(model_classification);
classLoss = kfoldLoss(model_cross);
crossval_elapsed = toc(crossval_start);

% Results
fprintf('SVM linear training done in: %f seconds.\n',ctrain_elapsed);
fprintf('10-fold cross-validation done in: %f seconds.\n',crossval_elapsed);
fprintf('Accuracy: %f\n\n',1-classLoss);

figure
gscatter(data_all(:,1),data_all(:,2),data_all(:,6))
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('occu','not','Support Vector')
hold off

%% task2 - Brute Force
param_grid.c = 10.^(-5:5);

if TASK2_KF == "rbf"
    param_grid.sigma = 10.^(-5:5);
elseif TASK2_KF == "polynomial"
    param_grid.q = (2:5);
else
    error("Invalid kernel function");
end

task2_start = tic;
[optimise_hyperparameters, op_stats, opt_acc] = GridSearchCV(data_all(:, 1:5), data_all(:,6), param_grid, TASK2_KF, K_FOLD);
tast2_end = toc(task2_start);

fprintf("Optimisation of hyper-parameter done in: %f seconds.\n",tast2_end);
for i = 1 : length(op_stats)
    fprintf("Combination: %d | c: %f | sigma: %f | acc: %f\n", i, op_stats(i).c, op_stats(i).sigma, op_stats(i).acc);
end
fprintf("\nOptimise Combition is c: %f, sigma/q: %f\n", optimise_hyperparameters(1,1), optimise_hyperparameters(1,2));