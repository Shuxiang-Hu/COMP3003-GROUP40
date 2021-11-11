

function [x,y] = preprocess(data)
%PREPROCESS Preprocess data from the regression problem
%  DATA = LOAD_DATA(DATA) preprocess data for training
%  Read in data as a table. Return data as tables.
%
%  Example:
%  x,y = importfile(data_table);
%
  fprintf("Start processing data\n");
  data(any(ismissing(data),2),:) = [];
  ir = randperm(size(data,1));
  data = data(ir,:);
  x = data(:,{'x','x1','x2','x3','x4'});  
  y = data(:,{'stop'});
  fprintf("End of data preprocessing\n");
end