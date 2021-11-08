%% 
clear;
clc;

%% useful pre-set data

file_name = 'datatraining.txt';

num_attributes = 5;
num_labels = 2;
num_sets = 1;
num_lines = 1000; % count of training data

features_training = zeros(num_lines, num_attributes);

%% extract features and labels in training dataset and test dataset

training_file = importdata(file_name).data;
label1 = training_file(:,6) == 1;
label0 = training_file(:,6) == 0;

l1_set = training_file(label1,:);
l0_set = training_file(label0,:);

%% Normalisation
normal0 = mapminmax(l0_set(:, 1:5)', 0, 1);
normal1 = mapminmax(l1_set(:, 1:5)', 0, 1);

%% Correlation analysis
corr = corrcoef([normal0 normal1]');
% feature 2 and 5 are highly correlated
normal0 = normal0(1:4, :);
normal1 = normal1(1:4, :);

%% Write datasets
writematrix([normal0' l0_set(:, 6)], 'datalabel0.txt');
writematrix([normal1' l1_set(:, 6)], 'datalabel1.txt');