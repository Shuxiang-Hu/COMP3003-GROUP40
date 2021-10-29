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

% set up cross-validation parameters
k_fold = 5;

% set up result (RMSE) matrix
sigma_num = 10;
epsilon_num = 10;
rmses = zeros(sigma_num, epsilon_num);

% hyperparameter tuning 
i = 1;
for sigma = 0.1:0.1:1
    j = 1;
    for epsilon = 0.1:0.1:1
        % train model and cross-validating
        mdlRbf = fitrsvm(x_train, y_train, 'KernelFunction','rbf', 'KernelScale', sigma, 'BoxConstraint', 1 , 'Standardize', true, 'KFold', k_fold, 'Epsilon', 0.1);
        mdlRbf.Trained;
        
        % calculate the average rmse
        rmseRbf = sqrt(kfoldLoss(mdlRbf));

        % record the result
        rmses(i, j) = rmseRbf;
    
        j = j + 1;
    end
    i = i + 1;
end