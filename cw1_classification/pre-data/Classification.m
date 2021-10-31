clear;
clc;

%% Pre-processing
data_all = importdata("datatraining.txt");
data_processed = data_all.data;

%% Classification model
ctrain_start = tic;
model_classification = fitcsvm(data_processed(1:1000,1:5),data_processed(1:1000,6), 'KernelFunction','linear', 'BoxConstraint',1);
ctrain_elapsed = toc(ctrain_start);

%% 10-fold cross-validation
crossval_start = tic;
model_cross = crossval(model_classification);
classLoss = kfoldLoss(model_cross);
crossval_elapsed = toc(crossval_start);

%% Results
fprintf('Training done in: %f seconds.\n',ctrain_elapsed);
fprintf('Cross-validation done in: %f seconds.\n',crossval_elapsed);
fprintf('Accuracy: %f\n',1-classLoss);
