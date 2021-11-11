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

%% Task1 Linear model with kfold (Task3)
task1_start = tic;
fold = 10;
x = data_all(:, 1:4);
y = data_all(:,5);
num_instances = height(x);
fold_size = floor(num_instances/fold);
fold_start = 1;
fold_end = fold_size;

mean_confusion_matrix = [0,0;0,0];
mean_accuracy = 0;
for k = 1:fold
        
        % split test and train
        x_test = x(fold_start:fold_end,:);
        y_test = y(fold_start:fold_end,:);
        x_train = x;
        x_train(fold_start:fold_end,:) = [];
        y_train = y;
        y_train(fold_start:fold_end,:) = [];
        model = fitcsvm(data_all(:,1:4),data_all(:,5), 'KernelFunction',TASK1_KF, 'BoxConstraint',1);
        
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
        mean_accuracy = acc+mean_accuracy
        mean_confusion_matrix  = mean_confusion_matrix + [TN,FN;FP,TP];
        fprintf("Generalized accuracy on %dth folder: %f\n", k,acc);
        fprintf("Confusion Matrix for fold %d: \n",k);
        fprintf("TN:%d FN:%d\n",TN,FN);
        fprintf("FP:%d TP:%d\n",FP,TP);
        
        fold_start = fold_start + fold_size;
        fold_end = fold_end + fold_size;
        if k == fold - 1
            fold_end = num_instances;
        end
        
    end
task1_elapsed = toc(task1_start);

fprintf("Linear SVM training done in: %f seconds.\n",task1_elapsed);
mean_confusion_matrix = mean_confusion_matrix/fold
mean_accuracy = mean_accuracy / fold

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



