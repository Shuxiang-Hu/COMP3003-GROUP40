
function [hyperparameter_stats, opt_hyperparameters, opt_rmse] = innerCV(x_train, y_train, kernel_method, param_grid, k_fold)
    
    dataset_size = size(x_train, 1);
    param_size = size(param_grid, 1);
    
    % pre-allocate the memory for op_stats
    if strcmp(kernel_method, "rbf")
        hyperparameter_stats = struct("sigma", zeros(1, param_size), "c", zeros(1, param_size), ...
            "epsilon", zeros(1,param_size), "sv_stats", zeros(2,k_fold,param_size));
    elseif strcmp(kernel_method, "polynomial") 
        hyperparameter_stats = struct("q", zeros(1, param_size), "c", zeros(1,param_size), ...
            "epsilon", zeros(1,param_size), "sv_stats", zeros(2,k_fold,param_size));
    else
        error("Invalid kernel method");
    end

    % set up optimal_hyperparameters matrix
    % row - the different hp combination
    % col - either sigma, box constraint and epsilon or q, box constraint 
    %       and epsilon
    opt_hyperparameters = zeros(1, 3);
    
    % initialize the best RMSE to be infinite
    opt_rmse = Inf;

    % hyperparameter tuning 
    for i = 1:size(param_grid, 1)

        % initialize the RMSE matrix for this set of hyperparameters
        rmses = zeros(1, k_fold);

        % initialize ending index for the validation set
        validation_end = 0;
       
        % check for kernel method type and record current hyperparameter
        if strcmp(kernel_method, 'rbf')
            hyperparameter_stats(i).sigma = param_grid(i, 1);
        end
        if strcmp(kernel_method, 'polynomial') 
            hyperparameter_stats(i).q = param_grid(i, 1);
        end
        hyperparameter_stats(i).c = param_grid(i, 2);
        hyperparameter_stats(i).epsilon = param_grid(i, 3);
        
        sv_stats = zeros(k_fold,2);

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
            if strcmp(kernel_method, 'rbf')
                mdl = fitrsvm(x_train_set, y_train_set, 'KernelFunction','rbf', 'KernelScale', param_grid(i, 1), 'BoxConstraint', param_grid(i, 2), 'Epsilon', param_grid(i, 3), 'Standardize', true);
            end
            if strcmp(kernel_method, 'polynomial') 
                mdl = fitrsvm(x_train_set, y_train_set, 'KernelFunction','polynomial', 'PolynomialOrder', param_grid(i, 1), 'BoxConstraint', param_grid(i, 2), 'Epsilon', param_grid(i, 3), 'Standardize', true);
            end
            
            % evaluate the model by predicting on the validation set
            y_predict = mdl.predict(x_validation_set);
            rmses(1, j) = sqrt(mean(y_predict - table2array(y_validation_set)).^2);

            % record the number and percentage of the support vectors for each
            % model
            sv_stats(j, 1) = length(mdl.SupportVectors);
            sv_stats(j, 2) = sv_stats(1)/size(x_train_set, 1);

        end
        % record suppor vector stat to the corresponding parameter set
        hyperparameter_stats(i).sv_stats = sv_stats;

        % calculate the average RMSE for this set of hyperparameters
        average_rmse = mean(rmses);

        % record the result if the model is better than previous
        if(average_rmse<=opt_rmse)
            opt_hyperparameters(:) = param_grid(i, :);
            opt_rmse = average_rmse;
        end
    end
end