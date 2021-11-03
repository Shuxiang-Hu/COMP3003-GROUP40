
function [support_vector_num, optimal_hyperparameters, min_rmse] = GridSearchCV(x_train, y_train, kernel_method, hyperparameters, k_fold)
    
    % set up support_vector_num and optimal_hyperparameters matrix, 
    % row - the different hp combination
    % col - either sigma, box constraint and epsilon or q, box constraint 
    %       and epsilon
    support_vector_num = zeros(size(hyperparameters, 1), k_fold);
    optimal_hyperparameters = zeros(1, 3);
    
    % initialize the best RMSE to be infinite
    min_rmse = Inf;

    % set up the initial ending index for the validation set
    validation_end = 0;

    % set up the dataset size (how many observations)
    dataset_size = size(y_train, 1);
    
    % hyperparameter tuning 
    for i = 1:size(hyperparameters, 1)

        % initialize the RMSE matrix for this set of hyperparameters
        rmses = zeros(1, k_fold);

        % inner cross-validation
        for j = 1:k_fold
            validation_start = validation_end + 1;
            validation_end = round(dataset_size * j/k_fold);
            
            % split the dataset into training and validation
            x_validation_set = x_train(validation_start:validation_end, :);
            x_train_set = x_train(~ismember(1:dataset_size, (validation_start:validation_end)), :);
            y_validation_set = y_train(validation_start:validation_end, :);
            y_train_set = y_train(~ismember(1:dataset_size, (validation_start:validation_end)), :);
            
            % check for kernel method type and train the SVM regression model
            if strcmp(kernel_method, 'gaussian_rbf')
                mdl = fitrsvm(x_train_set, y_train_set, 'KernelFunction','rbf', 'KernelScale', hyperparameters(i, 1), 'BoxConstraint', hyperparameters(i, 2), 'Epsilon', hyperparameters(i, 3));
            end
            if strcmp(kernel_method, 'polynomial') 
                mdl = fitrsvm(x_train_set, y_train_set, 'KernelFunction','polynomial', 'PolynomialOrder', hyperparameters(i, 1), 'BoxConstraint', hyperparameters(i, 2), 'Epsilon', hyperparameters(i, 3));
            end
            
            % evaluate the model by predicting on the validation set
            y_predict = mdl.predict(x_validation_set);
            rmses(1, j) = sqrt(mean(y_predict - table2array(y_validation_set)).^2);

            % record the number and percentage of the support vectors for each
            % model
            support_vector_num(i, j, 1) = length(mdl.SupportVectors);
            support_vector_num(i, j, 2) = support_vector_num(i, j, 1)/dataset_size;
        end

        % calculate the average RMSE for this set of hyperparameters
        average_rmse = mean(rmses);

        % record the result if the model is better than previous
        if(average_rmse<=min_rmse)
            optimal_hyperparameters(:) = hyperparameters(i, :);
            min_rmse = average_rmse;
        end
    end
end