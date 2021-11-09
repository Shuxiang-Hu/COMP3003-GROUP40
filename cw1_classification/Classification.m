clear;
clc;

%% Load data and Related element
% pre-set element


NUM_LABEL0_DATA = 1900;
NUM_LABEL1_DATA = 1900;

K_FOLD = 10;

% kernel function
TASK1_KF = "linear";
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

%% Task1 Linear model
task1_start = tic;
model_classification = fitcsvm(data_all(:,1:4),data_all(:,5), 'KernelFunction',TASK1_KF, 'BoxConstraint',1);
task1_elapsed = toc(task1_start);

fprintf("Linear SVM training done in: %f seconds.\n",task1_elapsed);


%% Task2 - Brute Force
% parameter range setting

param_range.c = 2.^(-3:3);
kfold = 10;
lfold = 10;
x = data_all(:, 1:4);
y = data_all(:,5);

mean_confusion_matrix = [0,0;0,0];

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
            elseif strcmp(TASK2_KF, "polynomial") 
                model = fitcsvm(x_train, y_train, "KernelFunction", TASK2_KF, "PolynomialOrder", optimise_hyperparameters(1), "BoxConstraint", optimise_hyperparameters(2));
            end

        % predict and evaluate
        y_pre = model.predict(x_test);
        
        % compute the confusion matrix
        %         actual  0     1
        % predict         
        %       0         TN    FN
        %       1         FP    TP
        temp = y_pre+y_test;
        TN = sum(temp == 0);
        TP = sum(temp ==2);
        temp = y_pre - y_test;
        FP = sum(temp == 1);
        FN = sum(temp == -1);
        acc = (TP+TN)/(TP+TN+FP+FN);
        mean_confusion_matrix  = mean_confusion_matrix + [TN,FN;FP,TP];
        fprintf("Generalized accuracy on %dth folder: %f\n", k,acc);
        fprintf("Confusion Matrix for fold %d: \n",k);
        fprintf("TN:%d FN:%d\n",TN,FN);
        fprintf("FP:%d TP:%d\n",FP,TP);
        
        if strcmp(TASK2_KF, "rbf")
            for i = 1 : length(op_stats)
                fprintf("Combination: %d | c: %f | sigma: %f | acc: %f\n", i, op_stats(i).c, op_stats(i).sigma, op_stats(i).acc);
            end
        elseif strcmp(TASK2_KF, "polynomial") 
            for i = 1 : length(op_stats)
                fprintf("Combination: %d | c: %f | sigma: %f | acc: %f\n", i, op_stats(i).c, op_stats(i).q, op_stats(i).acc);
            end
        end
        fprintf("Optimise Combition is c: %f, sigma/q: %f\n\n", optimise_hyperparameters(1,2), optimise_hyperparameters(1,1));
        fold_start = fold_start + fold_size;
        fold_end = fold_end + fold_size;
        if k == kfold - 1
            fold_end = num_instances;
        end
        
    end
tast2_end = toc(task2_start);
fprintf("Optimisation of hyper-parameter done in: %f seconds.\n",tast2_end);
fprintf("Average Confusion Matrix over the folds:\n ");
mean_confusion_matrix/kfold


