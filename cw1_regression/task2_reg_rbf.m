
% modification required after data pre-processing's completion 
% for testing purpose, only the first 3000 data is used
x = E0C3000N5Cov(2:end, 2:end);
y = E0C3000N5Cov(2:end, 1);

% modification required for outer cross-validation
% currently set the first 600 data to be the test data
x_train = x(601:end, :);
x_test = x(1:600, :);
y_train = y(601:end, :);
y_test = y(1:600, :);
train_data_size = 2400;

% set up cross-validation parameters
k_fold = 5;

% set up result (RMSE) matrix
min_sigma = 0.1;
max_sigma = 1;
sigma_interval = 0.1;

min_epsilon = 0.1;
max_epsilon = 1;
epsilon_interval = 0.1;

sigma_num = max_sigma/sigma_interval;
epsilon_num = max_epsilon/epsilon_interval;
rmses = zeros(sigma_num, epsilon_num);

% set up support_vector_num matrix
support_vector_num = zeros(sigma_num, epsilon_num);

% hyperparameter tuning 
i = 1;
for sigma = 0.1:0.1:1
    j = 1;
    for epsilon = 0.1:0.1:1
        % train model and cross-validating
        mdlRbf = fitrsvm(x_train, y_train, 'KernelFunction','rbf', 'KernelScale', sigma, 'BoxConstraint', 1 , 'Standardize', true, 'KFold', k_fold, 'Epsilon', 0.1);
        
        % record the number and percentage of the support vectors for each
        % model
        for n = 1:k_fold
            support_vector_num(i, j, n, 1) = length(mdlRbf.Trained{n, 1}.SupportVectors);
            support_vector_num(i, j, n, 2) = support_vector_num(i, j, n, 1)/train_data_size;
        end
        
        % calculate the average rmse
        rmse = sqrt(kfoldLoss(mdlRbf));

        % record the result
        rmses(i, j) = rmse;
    
        j = j + 1;
    end
    i = i + 1;
end

% search for the hyperparameters case with min RMSE 
minRmse = min(min(rmses));
[i, j] = find(rmses==minRmse);

% retrive the sigma and epsilon
best_sigma = min_sigma * i(1);
best_epsilon = min_epsilon * j(1);