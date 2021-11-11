
function [model,rmses] = train(x,y,kfold,lfold,kernel_method,param_grid)
    % perform k fold cross validation
    
    rmses = zeros(1, kfold); %rmes over kfolds
    num_instances = height(x);
    fold_size = floor(num_instances/kfold);
    fold_start = 1;
    fold_end = fold_size;
    
    for k = 1:kfold
        
        % split test and train
        x_test = x(fold_start:fold_end,:);
        y_test = y(fold_start:fold_end,:);
        x_train = x;
        x_train(fold_start:fold_end,:) = [];
        y_train = y;
        y_train(fold_start:fold_end,:) = [];
        
        % perform inner corssvalidation to find good hp cobination
        [support_vector_num, optimal_hyperparameters, min_rmse] = innerCV(x_train, y_train, kernel_method, param_grid, lfold);
        
        % train a model for current fold with good parameters found
        if strcmp(kernel_method, "rbf")
            model = fitrsvm(x_train, y_train, "KernelFunction", kernel_method, "KernelScale", optimal_hyperparameters(1), "BoxConstraint", optimal_hyperparameters(2),'Epsilon', optimal_hyperparameters(3));
        elseif strcmp(kernel_method ,"polynomial") 
            model = fitrsvm(x_train, y_train, "KernelFunction", kernel_method, "PolynomialOrder", optimal_hyperparameters(1), "BoxConstraint", optimal_hyperparameters(2),'Epsilon', optimal_hyperparameters(3));
        end
        % predict and evaluate
        y_pre = model.predict(x_test);
        rmses(1, k) = sqrt(mean(y_pre - table2array(y_test)).^2);
        fprintf("RMSE of %dth fold: %f\n",k,rmses(1,k));
        
        fold_start = fold_start + fold_size;
        fold_end = fold_end + fold_size;
        if k == kfold - 1
            fold_end = num_instances;
        end
    end
end