%% 
clear;
clc;

%% useful pre-set data

training_data_name = 'datatraining';
test_data_name = 'datatest';

fprintf(['Process dataset: ', training_data_name, ' original data\n']);
fprintf(['Process dataset: ', test_data_name, ' original data\n']);
training_set = [training_data_name, '.txt'];
test_set = [test_data_name, '.txt'];

num_training_attributes = 6;
num_training_labels = 2;
num_training_sets = 1;
num_training_lines = 1000;

features_training = zeros(num_lines, num_attributes);
labels_training = zeros(1, num_lines);

num_test_attributes = 6;
num_test_labels = 2;
num_test_sets = 1;
num_test_lines = 200;

features_test = zeros(num_lines, num_attributes);
labels_test = zeros(1, num_lines);

%% processing the training dataset

training_file = fopen(training_set);
if f == -1
    error('Fail to open file %s#n', training_set);
end

%% processing the test dataset

test_file = fopen(test_set);
if f == -1
    error('Fail to open file %s#n', training_set);
end

