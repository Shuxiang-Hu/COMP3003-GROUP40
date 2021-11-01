% brute force
% param : features - 
%         labels - 
%         op_stats - the range of hyper parameters 
%         kernal_method - 'rbf' or 'polynomial'
%         k_fold -
function [optimise_hyperparameters, op_stats, opt_rmse] = GridSearchCV(features, labels, param_grid, kernel_method, k_fold)

    % sigma for RBF and q for the polynomial kernel; box-constraint C
    optimise_hyperparameters = zeros(1,2);
    if kernel_method == "rbf"
        [c, sigma] = ndgrid(param_grid.c, param_grid.sigma);
        hp_combination_set = [c(:) sigma(:)];
        num_combination = size(hp_combination_set, 1);
        op_stats = struct("c", zeros(1,num_combination), "sigma", zeros(1,num_combination),...
            "rmse", zeros(1,num_combination), "sv_stats", zeros(2,k_fold,num_combination));
    elseif kernel_method == "polynomial"
        [c, q] = ndgrid(param_grid.c, param_grid.q);
        hp_combination_set = [c(:) q(:)];
        num_combination = size(hp_combination_set, 1);
        op_stats = struct("c", zeros(1,num_combination), "q", zeros(1,num_combination),...
            "rmse", zeros(1,num_combiation), "sv_stats", zeros(2,k_fold,num_combination));
    else
        error("Invalid kernel method");
    end
    
    
    data_size = size(labels, 1);
    opt_rmse = Inf;

    for i = 1 : num_combination
        
        % Two different kernal functions: rbf and polynomial
        mdl = fitcsvm(features, labels, 'KernelFunction',kernel_method, 'KernelScale', hp_combination_set(i, 1), 'BoxConstraint', hp_combination_set(i, 2) , 'Standardize', true, 'KFold', k_fold);

        rmse = sqrt(kfoldLoss(mdl));
        
        if (rmse < opt_rmse)
            optimise_hyperparameters(:) = hp_combination_set(i, :);
            opt_rmse = rmse;
        end
        
        % number of support vector and percent of training data
        stats = zeros(k_fold,2);
        for k = 1 : k_fold
            stats(k,1) = length(mdl.Trained{k, 1}.SupportVectors);
            stats(k,2) = stats(k,1) / data_size;
        end
        
        op_stats(i).c = hp_combination_set(i, 1);
        if kernel_method == "rbf"
            op_stats(i).sigma = hp_combination_set(i, 2);
        elseif kernel_method == "polynomial"
            op_stats(i).q = hp_combination_set(i, 2);
        end
        op_stats(i).sv_stats = stats;
        op_stats(i).rmse = rmse;

    end
end