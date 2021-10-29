
x = E0C3000N5Cov(2:end, 2:end);
y = E0C3000N5Cov(2:end, 1);
x_train = x(601:end, :);
x_test = x(1:600, :);
y_train = y(601:end, :);
y_test = y(1:600, :);
k_fold = 5;
sigma_num = 10;
epsilon_num = 10;

rmses = zeros(sigma_num, epsilon_num);

i = 1;
for sigma = 0.1:0.1:1
    j = 1;
    for epsilon = 0.1:0.1:1
        MdlLin = fitrsvm(x_train, y_train, 'KernelFunction','rbf', 'KernelScale', sigma, 'BoxConstraint', 1 , 'Standardize', true, 'KFold', k_fold, 'Epsilon', 0.1);
        MdlLin.Trained;
        rmseLin = sqrt(kfoldLoss(MdlLin));
        rmses(i, j) = rmseLin;
    
        j = j + 1;
    end
    i = i + 1;
end