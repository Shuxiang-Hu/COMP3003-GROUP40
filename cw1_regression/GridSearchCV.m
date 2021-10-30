function [support_vector_num, optimal_hyperparameters, min_rmse] = GridSearchCV(x_train, y_train, kernel_method, hyperparameters, k_fold)
    
    % set up support_vector_num and optimal_hyperparameters matrix, 
    % row - the different hp combination
    % col - either sigma, box constraint and epsilon or q, box constraint 
    %       and epsilon
    support_vector_num = zeros(size(hyperparameters, 1), 3);
    optimal_hyperparameters = zeros(1, 3);
    
    % initialize the best RMSE to be infinite
    min_rmse = Inf;

    data_size = size(y_train, 1);
    
    % hyperparameter tuning 
    for i = 1:size(hyperparameters, 1)
        % check for kernel method type, train model and cross-validating
        if strcmp(kernel_method, 'gaussian_rbf')
            mdl = fitrsvm(x_train, y_train, 'KernelFunction','rbf', 'KernelScale', hyperparameters(i, 1), 'BoxConstraint', hyperparameters(i, 2) , 'Standardize', true, 'KFold', k_fold, 'Epsilon', hyperparameters(i, 3));
        end
        if strcmp(kernel_method, 'polynomial') 
            mdl = fitrsvm(x_train, y_train, 'KernelFunction','polynomial', 'PolynomialOrder', hyperparameters(i, 1), 'BoxConstraint', hyperparameters(i, 2) , 'Standardize', true, 'KFold', k_fold, 'Epsilon', hyperparameters(i, 3));
        end
        
        % record the number and percentage of the support vectors for each
        % model
        for n = 1:k_fold
            support_vector_num(i, n, 1) = length(mdl.Trained{n, 1}.SupportVectors);
            support_vector_num(i, n, 2) = support_vector_num(i, n, 1)/data_size;
        end
        
        % calculate the average rmse
        rmse = sqrt(kfoldLoss(mdl));

        % record the result
        if(rmse<=min_rmse)
            optimal_hyperparameters(:) = hyperparameters(i, :);
            min_rmse = rmse;
        end
    end
end