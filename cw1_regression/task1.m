
%%
clear all;
close all;
%% Data Preprocessing
filename = "data.csv";
dataLines = [2, Inf];
%% Set up the Import Options and import the data
data = loadData(filename, dataLines);
[x, y] = preprocess(data);

%% train model
kfold = 10;
 rmses = zeros(1, kfold); %rmes over kfolds
 num_instances = height(x);
 fold_size = floor(num_instances/kfold);
 fold_start = 1;
 fold_end = fold_size;
 mean_rmse = 0;
 mean_sup = 0;
for k = 1:kfold
        
    % split test and train
    x_test = x(fold_start:fold_end,:);
    y_test = y(fold_start:fold_end,:);
    x_train = x;
    x_train(fold_start:fold_end,:) = [];
    y_train = y;
    y_train(fold_start:fold_end,:) = [];
    eps = 0;
    model = fitrsvm(x_train, y_train, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Epsilon', eps);
     
    y_pre = model.predict(x_test);
    rmse = sqrt(mean(y_pre - table2array(y_test)).^2);

    %fprintf("RMSE of %dth fold: %f\n",k,rmse);
    fprintf("num of supvec of %dth fold: %f\n",k,length(model.SupportVectors));
    mean_sup = mean_sup + length(model.SupportVectors);
    mean_rmse = mean_rmse+rmse;
    fold_start = fold_start + fold_size;
    fold_end = fold_end + fold_size;
    if k == kfold - 1
        fold_end = num_instances;
    end
end
mean_rmse = mean_rmse/kfold;
mean_sup = mean_sup/kfold;
fprintf("Average rmse: %f\n",mean_rmse);
figure,plot(table2array(x_test(:,2)), table2array(y_test(:,1)),'or');
figure,plot(table2array(x_test(:,2)), y_pre(:,1),'or');

title("linear regression on original data");
