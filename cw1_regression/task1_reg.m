%%
clear all;
close all;
%% Data Preprocessing
filename = "70E_50C_3000N_5Cov.csv";
dataLines = [2, Inf];
%% Set up the Import Options and import the data
data = load_data(filename, dataLines);
[x, y] = preprocess(data);

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

title("linear regression on original data");





