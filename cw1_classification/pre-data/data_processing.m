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

writematrix(l0_set, 'datalabel0.txt');
writematrix(l1_set, 'datalabel1.txt');