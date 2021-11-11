


kernel_method = "polynomial";
%kernel_method = "rbf";
run("data.csv",10,10,kernel_method);

function run(file_path,kfold,lfold,kernel_method)
    data = loadData(file_path);
    [x,y] = preprocess(data);
    
    param_range.c = 2.^(0:3);
    param_range.ep = 2.^(0:3);
    if kernel_method == "rbf"
        param_range.sigma = 2.^(-3:0);
        [c, sigma,ep] = ndgrid(param_range.c, param_range.sigma,param_range.ep);
        param_grid = [sigma(:) c(:) ep(:)];
    elseif kernel_method == "polynomial"
        param_range.q = 2.^(-3:0);
        [c, q,ep] = ndgrid(param_range.c, param_range.q,param_range.ep);
        param_grid = [q(:) c(:) ep(:)];
    else
        error("Invalid kernel function");
    end
    
    [model,rmses] = train(x,y,kfold,lfold,kernel_method,param_grid);
    
    for i = 1:size(rmses,2)
        fprintf('rmse of %dth fold: %f\n',i,rmses(i) );
    end
end