



function run(filepath, kfold, lfold,kernel_method,hyper_parameters)
    data = loadData(filepath);
    x,y = preprocess(data);
    model = train(x,y,kfold,lfold,kernel_method,hyper_parameters);
end