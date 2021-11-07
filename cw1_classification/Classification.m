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

%% Task1 Linear model
task1_start = tic;
model_classification = fitcsvm(data_all(:,1:4),data_all(:,5), 'KernelFunction',TASK1_KF, 'BoxConstraint',1);
task1_elapsed = toc(task1_start);

fprintf("Linear SVM training done in: %f seconds.\n",task1_elapsed);


%% Task2 - Brute Force
% parameter range setting

param_range.c = 10.^(-3:3);

param_grid.c = 10.^(-5:5);
kfold = 10;
lfold = 10;
x = data_all(:, 1:4);
y = data_all(:,5);
acc = zeros(1,kfold);


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


    rmses = zeros(1, kfold); %rmes over kfolds
    num_instances = height(x);
    fold_size = floor(num_instances/kfold);
    fold_start = 1;
    fold_end = fold_size;
    
    for k = 1:kfold
        
        % split test and train
        x_test = x(fold_start:fold_end,:);
        y_test = y(fold_start:fold_end,:);
        x_train = x;
        x_train(fold_start:fold_end,:) = [];
        y_train = y;
        y_train(fold_start:fold_end,:) = [];
        [op_stats,optimise_hyperparameters,opt_acc] = innerCV(x_train, y_train, TASK2_KF,param_grid,  lfold);

        % train a model for current fold with good parameters found
            if strcmp(TASK2_KF, "rbf")
                model = fitcsvm(x_train, y_train, "KernelFunction", TASK2_KF, "KernelScale", optimise_hyperparameters(1), "BoxConstraint", optimise_hyperparameters(2));
            elseif strcmp(kernel_method, "polynomial") 
                model = fitcsvm(x_train, y_train, "KernelFunction", TASK2_KF, "PolynomialOrder", optimise_hyperparameters(1), "BoxConstraint", optimise_hyperparameters(2));
            end

        % predict and evaluate
        y_pre = model.predict(x_test);
        
        acc(1,k) = sum(y_pre == y_test)/size(y_pre,1);
        fprintf("Generalized accuracy on %dth folder: %f\n", k,acc(k));
        
        for i = 1 : length(op_stats)
            fprintf("Combination: %d | c: %f | sigma: %f | acc: %f\n", i, op_stats(i).c, op_stats(i).sigma, op_stats(i).sv_stats(1,2));
        end
        fprintf("\nOptimise Combition is c: %f, sigma/q: %f\n", optimise_hyperparameters(1,1), optimise_hyperparameters(1,2));
        fold_start = fold_start + fold_size;
        fold_end = fold_end + fold_size;
        if k == kfold - 1
            fold_end = num_instances;
        end
        
    end
tast2_end = toc(task2_start);
fprintf("Optimisation of hyper-parameter done in: %f seconds.\n",tast2_end);


