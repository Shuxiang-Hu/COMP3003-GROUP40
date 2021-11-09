%%
clear all;
close all;
%% Data Preprocessing
filename = "70E_50C_3000N_5Cov.csv";
dataLines = [2, Inf];
%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["nid", "status", "start", "stop", "z", "x", "x1", "x2", "x3", "x4"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
data = readtable(filename, opts);

data(any(ismissing(data),2),:) = [];
x = data(:,{'x','x1','x2','x3','x4'});  
y = data(:,{'stop'});

%% train model
x_test = x(1:300,:);
y_test = y(1:300,:);
x_train = x;
x_train(1:300,:) = [];
y_train = y;
y_train(1:300,:) = [];
        
eps = 1;
model = fitrsvm(x_train, y_train, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Epsilon', eps);
     
y_pre = model.predict(x_test);
rmses = sqrt(mean(y_pre - table2array(y_test)).^2);
      
figure,plot(table2array(x_test(:,2)), table2array(y_test(:,1)),'or');
figure,plot(table2array(x_test(:,2)), y_pre(:,1),'or');

hold on;plot(X_prediction,Y_prediction,'b');
title("linear regression on original data");





