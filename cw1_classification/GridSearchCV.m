% brute force
% param : features - 
%         labels - 
%         op_stats - the range of hyper parameters 
%         kernal_method - 'rbf' or 'polynomial'
%         k_fold -
function [optimise_hyperparameters, op_stats, opt_acc] = GridSearchCV(features, labels, param_grid, kernel_method, k_fold)

    % sigma for RBF and q for the polynomial kernel; box-constraint C
    optimise_hyperparameters = zeros(1,2);
    if kernel_method == "rbf"
        [c, sigma] = ndgrid(param_grid.c, param_grid.sigma);
        hp_combination_set = [c(:) sigma(:)];
        num_combination = size(hp_combination_set, 1);
        % pre-allocate the memory for op_stats
        op_stats = struct("c", zeros(1,num_combination), "sigma", zeros(1,num_combination),...
            "acc", zeros(1,num_combination), "sv_stats", zeros(3,k_fold,num_combination));
    elseif kernel_method == "polynomial"
        [c, q] = ndgrid(param_grid.c, param_grid.q);
        hp_combination_set = [c(:) q(:)];
        num_combination = size(hp_combination_set, 1);
        % pre-allocate the memory for op_stats
        op_stats = struct("c", zeros(1,num_combination), "q", zeros(1,num_combination),...
            "acc", zeros(1,num_combiation), "sv_stats", zeros(2,k_fold,num_combination));
    else
        error("Invalid kernel method");
    end
    
    data_size = size(labels, 1);
    opt_acc = 0;
    dpf = floor(data_size / k_fold);% data per folder

    for i = 1 : num_combination
        accs = zeros(1,k_fold);
        stats = zeros(k_fold,2);
        
        % cross validation
        for k = 1 : k_fold
            indice_start = dpf * (k - 1) + 1;
            indice_end = dpf * k;
            
            if k == k_fold
                indice_end = data_size;
            end
            
            % divide dataset into train and test subsets
            features_train = features;
            features_train(indice_start:indice_end,:) = [];
            labels_train = labels;
            labels_train(indice_start:indice_end,:) = [];
            features_test = features(indice_start:indice_end,:);
            labels_test = labels(indice_start:indice_end,:);
            
            % train the model
            mdl = fitcsvm(features_train, labels_train, 'KernelFunction',kernel_method, 'KernelScale', hp_combination_set(i, 1), 'BoxConstraint', hp_combination_set(i, 2) , 'Standardize', true);
            labels_predict = mdl.predict(features_test);
            
            accs(1,k) = length(find((labels_predict - labels_test) == 0))/length(labels_predict);
            
            stats(k,1) = length(mdl.SupportVectors);
            stats(k,2) = stats(k,1) / data_size;
            stats(k,3) = accs(1,k);
        end
        
        % update the minimize rmse
        ave_acc = mean(accs);
        if (ave_acc > opt_acc)
            optimise_hyperparameters(:) = hp_combination_set(i, :);
            opt_acc = ave_acc;
        end
        
        % write the data into return value
        op_stats(i).c = hp_combination_set(i, 1);
        if kernel_method == "rbf"
            op_stats(i).sigma = hp_combination_set(i, 2);
        elseif kernel_method == "polynomial"
            op_stats(i).q = hp_combination_set(i, 2);
        end
        op_stats(i).sv_stats = stats;
        op_stats(i).acc = ave_acc;

    end
end