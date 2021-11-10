%%%%%%%%%%%%%%%%%%%%%%%Data Processing%%%%%%%%%%%%%%%%%%%%%%%
% requires manually import the 70E_50C_3000N_5Cov.csv dataset 
x = E50C3000N5Cov(2:end, 2:end);
y = E50C3000N5Cov(2:end, 1);

% set the first 600 data to be the test data
x_train = x(601:end, :);
x_test = x(1:600, :);
y_train = y(601:end, :);
y_test = y(1:600, :);
train_data_size = 2400;

%%%%%%%%%%%%%%%%%%%%%%%Hyperparameter sets set-up%%%%%%%%%%%%%%%%%%%%%%%
cs = 2.^(-3:3);
qs = (2:1:10);
epsilons = 2.^(-3:3);
sigmas = 2.^(-3:3);
[A, B, C] = ndgrid(qs, cs, epsilons);
hyperparameters_rbf = [A(:) B(:) C(:)];

[a, b, c] = ndgrid(sigmas, cs, epsilons);
hyperparameters_poly = [a(:) b(:) c(:)];

%%%%%%%%%%%%%%%%%%%%%%%Model Training%%%%%%%%%%%%%%%%%%%%%%%
% k_fold is set to 5
[hyperparameter_stats_poly, opt_hyperparameters_poly, opt_rmse_poly] = innerCV(x_train, y_train, 'polynomial', hyperparameters_rbf, 5);
[hyperparameter_stats_rbf, opt_hyperparameters_rbf, opt_rmse_rbf] = innerCV(x_train, y_train, 'rbf', hyperparameters_poly, 5);

%%%%%%%%%%%%%%%%%%%%%%%Plotting - appart from the observing value, other two%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%are set to their optimal values within the given range%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%Polynomial q%%%%%%%%%%%%%%%%%%%%%%%
ind = 1;
yl = zeros(size(qs, 2), 1);
yr = zeros(size(qs, 2), 1);
for i = 1:size(hyperparameter_stats_poly, 2)
    if ind > size(qs, 2)
        break
    end
    if (hyperparameter_stats_poly(1, i).q == qs(ind)) && (hyperparameter_stats_poly(1, i).c == 0.125) && (hyperparameter_stats_poly(1, i).epsilon == 8)
        sv_sum = sum(hyperparameter_stats_poly(1, i).sv_stats);
        svn_avg = sv_sum(1)/5;
        yl(ind, 1) = svn_avg;

        svp_avg = sv_sum(2)/5;
        yr(ind, 1) = svp_avg;

        ind = ind + 1;
    end
end

yyaxis left;
plot(qs, yl);
ylabel("Average Support Vector Number");

yyaxis right;
plot(qs, yr);
ylabel("Average Support Vector Percentage");

xlabel("Kernel Scale (q)");
title("Average Support Vector VS. Kernel Scale (q)");

%%%%%%%%%%%%%%%%%%%%%%%Polynomial Box Constraint (c)%%%%%%%%%%%%%%%%%%%%%%%
ind = 1;
yl = zeros(size(cs, 2), 1);
yr = zeros(size(cs, 2), 1);
for i = 1:size(hyperparameter_stats_poly, 2)
    if ind > size(cs, 2)
        break
    end
    if (hyperparameter_stats_poly(1, i).c == cs(ind)) && (hyperparameter_stats_poly(1, i).q == 6) && (hyperparameter_stats_poly(1, i).epsilon == 8)
        sv_sum = sum(hyperparameter_stats_poly(1, i).sv_stats);
        svn_avg = sv_sum(1)/5;
        yl(ind, 1) = svn_avg;

        svp_avg = sv_sum(2)/5;
        yr(ind, 1) = svp_avg;

        ind = ind + 1;
    end
end

yyaxis left;
plot(cs, yl);
ylabel("Average Support Vector Number");

yyaxis right;
plot(cs, yr);
ylabel("Average Support Vector Percentage");

xlabel("Box Constraint (c)")
title("Average Support Vector VS. Box Constraint (c)")

%%%%%%%%%%%%%%%%%%%%%%%Polynomial Epsilon%%%%%%%%%%%%%%%%%%%%%%%
ind = 1;
yl = zeros(size(epsilons, 2), 1);
yr = zeros(size(epsilons, 2), 1);
for i = 1:size(hyperparameter_stats_poly, 2)
    if ind > size(epsilons, 2)
        break
    end
    if (hyperparameter_stats_poly(1, i).c == 0.125) && (hyperparameter_stats_poly(1, i).q == 6) && (hyperparameter_stats_poly(1, i).epsilon == epsilons(ind))
        sv_sum = sum(hyperparameter_stats_poly(1, i).sv_stats);
        svn_avg = sv_sum(1)/5;
        yl(ind, 1) = svn_avg;

        svp_avg = sv_sum(2)/5;
        yr(ind, 1) = svp_avg;

        ind = ind + 1;
    end
end
            
yyaxis left;
plot(epsilons, yl);
ylabel("Average Support Vector Number");

yyaxis right;
plot(epsilons, yr);
ylabel("Average Support Vector Percentage");

xlabel("Epsilon")
title("Average Support Vector VS. Epsilon")

%%%%%%%%%%%%%%%%%%%%%%%RBF Sigma%%%%%%%%%%%%%%%%%%%%%%%
ind = 1;
yl = zeros(size(sigmas, 2), 1);
yr = zeros(size(sigmas, 2), 1);
for i = 1:size(hyperparameter_stats_rbf, 2)
    if ind > size(sigmas, 2)
        break
    end
    if (hyperparameter_stats_rbf(1, i).sigma == sigmas(ind)) && (hyperparameter_stats_rbf(1, i).c == 8) && (hyperparameter_stats_rbf(1, i).epsilon == 8)
        sv_sum = sum(hyperparameter_stats_rbf(1, i).sv_stats);
        svn_avg = sv_sum(1)/5;
        yl(ind, 1) = svn_avg;

        svp_avg = sv_sum(2)/5;
        yr(ind, 1) = svp_avg;

        ind = ind + 1;
    end
end

yyaxis left;
plot(sigmas, yl);
ylabel("Average Support Vector Number");

yyaxis right;
plot(sigmas, yr);
ylabel("Average Support Vector Percentage");

xlabel("Sigma");
title("Average Support Vector VS. Sigma");


%%%%%%%%%%%%%%%%%%%%%%%RBF Box Constraint (c)%%%%%%%%%%%%%%%%%%%%%%%
ind = 1;
yl = zeros(size(cs, 2), 1);
yr = zeros(size(cs, 2), 1);
for i = 1:size(hyperparameter_stats_rbf, 2)
    if ind > size(cs, 2)
        break
    end
    if (hyperparameter_stats_rbf(1, i).c == cs(ind)) && (hyperparameter_stats_rbf(1, i).sigma == 0.125) && (hyperparameter_stats_rbf(1, i).epsilon == 8)
        sv_sum = sum(hyperparameter_stats_rbf(1, i).sv_stats);
        svn_avg = sv_sum(1)/5;
        yl(ind, 1) = svn_avg;

        svp_avg = sv_sum(2)/5;
        yr(ind, 1) = svp_avg;

        ind = ind + 1;
    end
end

yyaxis left;
plot(cs, yl);
ylabel("Average Support Vector Number");

yyaxis right;
plot(cs, yr);
ylabel("Average Support Vector Percentage");

xlabel("Box Constraint (c)")
title("Average Support Vector VS. Box Constraint (c)")

%%%%%%%%%%%%%%%%%%%%%%%RBF Epsilon%%%%%%%%%%%%%%%%%%%%%%%
ind = 1;
yl = zeros(size(epsilons, 2), 1);
yr = zeros(size(epsilons, 2), 1);
for i = 1:size(hyperparameter_stats_rbf, 2)
    if ind > size(epsilons, 2)
        break
    end
    if (hyperparameter_stats_rbf(1, i).c == 8) && (hyperparameter_stats_rbf(1, i).sigma == 0.125) && (hyperparameter_stats_rbf(1, i).epsilon == epsilons(ind))
        sv_sum = sum(hyperparameter_stats_rbf(1, i).sv_stats);
        svn_avg = sv_sum(1)/5;
        yl(ind, 1) = svn_avg;

        svp_avg = sv_sum(2)/5;
        yr(ind, 1) = svp_avg;

        ind = ind + 1;
    end
end
            
yyaxis left;
plot(cs, yl);
ylabel("Average Support Vector Number");

yyaxis right;
plot(cs, yr);
ylabel("Average Support Vector Percentage");

xlabel("Epsilon")
title("Average Support Vector VS. Epsilon")

title("Average Support Vector VS. Epsilon")