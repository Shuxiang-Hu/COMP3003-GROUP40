% brute force
% param : x_train - features
%         y_train - labels
%         kernal_method - "rbf" or "polynomial"
%         param_grid - the combination of hyperparameters
%         k_fold -
function [hyperparameter_stats, optimise_hyperparameters, opt_acc] = innerCV(x_train, y_train, kernel_method, param_grid, k_fold)

    % sigma for RBF and q for the polynomial kernel; box-constraint C
    optimise_hyperparameters = zeros(1,2);
    param_size = size(param_grid, 1);
    
    % pre-allocate the memory for op_stats
    if strcmp(kernel_method, "rbf")
        hyperparameter_stats = struct("sigma", zeros(1,param_size), "c", zeros(1,param_size), ...
            "sv_stats", zeros(2,k_fold,param_size), "acc", zeros(1, param_size));
    elseif strcmp(kernel_method, "polynomial") 
        hyperparameter_stats = struct("q", zeros(1,param_size), "c", zeros(1,param_size), ...
            "sv_stats", zeros(2,k_fold,param_size), "acc", zeros(1, param_size));
    else
        error("Invalid kernel method");
    end
    
    data_size = size(y_train, 1);
    opt_acc = 0;
    dpf = floor(data_size / k_fold);% data per folder

    for i = 1 : param_size
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
            x_train_set = x_train;
            x_train_set(indice_start:indice_end,:) = [];
            y_train_set = y_train;
            y_train_set(indice_start:indice_end,:) = [];
            x_validation_set = x_train(indice_start:indice_end,:);
            y_validation_set = y_train(indice_start:indice_end,:);
            
            % train the model
            if strcmp(kernel_method, "rbf")
                mdl = fitcsvm(x_train_set, y_train_set, "KernelFunction", kernel_method, "KernelScale", param_grid(i, 1), "BoxConstraint", param_grid(i, 2));
            elseif strcmp(kernel_method, "polynomial") 
                mdl = fitcsvm(x_train_set, y_train_set, "KernelFunction", kernel_method, "PolynomialOrder", param_grid(i, 1), "BoxConstraint", param_grid(i, 2));
            end
            labels_predict = mdl.predict(x_validation_set);
            accs(1,k) = length(find(labels_predict == y_validation_set))/length(labels_predict);
           
            stats(k,1) = length(mdl.SupportVectors);
            stats(k,2) = stats(k,1) / size(x_train_set, 1);
        end
        
        % update the minimize accuracy
        ave_acc = mean(accs);
        if (ave_acc > opt_acc)
            optimise_hyperparameters(:) = param_grid(i, :);
            opt_acc = ave_acc;
        end
        
        % write the data into return value
        hyperparameter_stats(i).c = param_grid(i, 2);
        if strcmp(kernel_method, "rbf")
            hyperparameter_stats(i).sigma = param_grid(i, 1);
        elseif strcmp(kernel_method, "polynomial")
            hyperparameter_stats(i).q = param_grid(i, 1);
        end
        hyperparameter_stats(i).sv_stats = stats;
        hyperparameter_stats(i).acc = ave_acc;
      
    end
end