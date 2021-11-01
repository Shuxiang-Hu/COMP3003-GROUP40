% random
% param : features - 
%         labels - 
%         param_distributions - limitation of hyperparameters (must)
%         n_iter - number of iteration
%         kernal_method - 'rbf' or 'polynomial'
%         k_fold - 
function [optimise_hyperparameters, support_vector_stats, opt_rmse] = RandomizedSearchCV(features, labels, param_distributions, n_iter, kernal_method, k_fold)

    % same to GridSearchCV
    optimise_hyperparameters = zeros(1,2);
    support_vector_stats = zerors(n_iter, k_fold);
    
    opt_rmse = Inf;
    data_size = size(labels, 1);

    for i = 1 : n_iter
        % Random generate C and sigma for RBF or q for the polynomial kernel
        
        mdl = fitcsvm(features, labels, 'KernelFunction',kernal_method, 'KernelScale', hyperparameters(i, 1), 'BoxConstraint', hyperparameters(i, 2) , 'Standardize', true, 'KFold', k_fold);

        for k = 1 : k_fold
            support_vector_stats(i, k, 1) = length(mdl.Trained{n, 1}.SupportVectors);
            support_vector_stats(i, k, 2) = support_vector_stats(i, k, 1) / data_size;
        end
        
        rmse = sqrt(kfoldLoss(mdl));
        
        if (rmse <= opt_rmse)
            optimise_hyperparameters(:) = hyperparameters(i, :);
            opt_rmse = rmse;
        end
    end
end