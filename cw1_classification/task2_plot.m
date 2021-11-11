clear;
clc;

% pre-set element
NUM_LABEL0_DATA = 2000;
NUM_LABEL1_DATA = 2000;

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

% parameter range setting
param_range.c = [0.01:0.03:3];
kfold = 10;
lfold = 10;
x = data_all(:, 1:4);
y = data_all(:,5);
acc = zeros(1,kfold);


if TASK2_KF == "rbf"
    param_range.sigma = [1];
    [c, sigma] = ndgrid(param_range.c, param_range.sigma);
    param_grid = [sigma(:) c(:)];
elseif TASK2_KF == "polynomial"
    param_range.q = [10];
    [c, q] = ndgrid(param_range.c, param_range.q);
    param_grid = [q(:) c(:)];
else
    error("Invalid kernel function");
end

[op_stats, optimise_hyperparameters, opt_acc] = innerCV(data_all(:, 1:4), data_all(:,5), TASK2_KF, param_grid, K_FOLD);

[~, col] = size(op_stats);
stats_mat = zeros(col, 5);

for i = 1 : col
    stats_mat(i,1) = op_stats(i).c;
    if TASK2_KF == "rbf"
        stats_mat(i,2) = op_stats(i).sigma;
    elseif TASK2_KF == "polynomial"
        stats_mat(i,2) = op_stats(i).q;
    end
    ave_sv_stats = mean(op_stats(i).sv_stats);
    stats_mat(i,3) = ave_sv_stats(1);
    stats_mat(i,4) = ave_sv_stats(2);
    stats_mat(i,5) = op_stats(i).acc;
end

yyaxis left
title("Poly:The number of support vector and a% with the increasing of c, where q = 10")
plot(stats_mat(:,1), stats_mat(:,3), '-o');
xlabel('q');
ylabel('Number of support vector');

yyaxis right
plot(stats_mat(:,1), stats_mat(:,4), '-x');
ylabel('a%');