function [] = train(x,y,kfold,lfold,kernel_method,param_grid)
    num_instances = height(x);
    fold_size = floor(num_instances/kfold);
    fold_start = 1;
    fold_end = fold_size;
    mean_confusion_matrix = [0,0;0,0];
    for k = 1:kfold
        
        % split test and train
        x_test = x(fold_start:fold_end,:);
        y_test = y(fold_start:fold_end,:);
        x_train = x;
        x_train(fold_start:fold_end,:) = [];
        y_train = y;
        y_train(fold_start:fold_end,:) = [];
        [op_stats,optimise_hyperparameters,opt_acc] = innerCV(x_train, y_train, kernel_method,param_grid,  lfold);

        % train a model for current fold with good parameters found
            if strcmp(kernel_method, "rbf")
                model = fitcsvm(x_train, y_train, "KernelFunction", kernel_method, "KernelScale", optimise_hyperparameters(1), "BoxConstraint", optimise_hyperparameters(2));
            elseif strcmp(kernel_method, "polynomial") 
                model = fitcsvm(x_train, y_train, "KernelFunction", kernel_method, "PolynomialOrder", optimise_hyperparameters(1), "BoxConstraint", optimise_hyperparameters(2));
            end

        % predict and evaluate
        y_pre = model.predict(x_test);
        
        % compute the confusion matrix
        %         actual  0     1
        % predict         
        %       0         TN    FN
        %       1         FP    TP
        temp = y_pre+y_test;
        TN = sum(temp == 0);
        TP = sum(temp ==2);
        temp = y_pre - y_test;
        FP = sum(temp == 1);
        FN = sum(temp == -1);
        acc = (TP+TN)/(TP+TN+FP+FN);
        mean_confusion_matrix  = mean_confusion_matrix + [TN,FN;FP,TP];
        fprintf("Generalized accuracy on %dth folder: %f\n", k,acc);
        fprintf("Confusion Matrix for fold %d: \n",k);
        fprintf("TN:%d FN:%d\n",TN,FN);
        fprintf("FP:%d TP:%d\n",FP,TP);
        
        if strcmp(kernel_method, "rbf")
            for i = 1 : length(op_stats)
                fprintf("Combination: %d | c: %f | sigma: %f | acc: %f\n", i, op_stats(i).c, op_stats(i).sigma, op_stats(i).acc);
            end
        elseif strcmp(kernel_method, "polynomial") 
            for i = 1 : length(op_stats)
                fprintf("Combination: %d | c: %f | sigma: %f | acc: %f\n", i, op_stats(i).c, op_stats(i).q, op_stats(i).acc);
            end
        end
        fprintf("Optimise Combition is c: %f, sigma/q: %f\n\n", optimise_hyperparameters(1,2), optimise_hyperparameters(1,1));
        fold_start = fold_start + fold_size;
        fold_end = fold_end + fold_size;
        if k == kfold - 1
            fold_end = num_instances;
        end
        
    end
    fprintf("Average Confusion Matrix over the folds:\n ");
    mean_confusion_matrix = mean_confusion_matrix/kfold
end