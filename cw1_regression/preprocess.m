

function [x,y] = preprocess(data)
%PREPROCESS Preprocess data from the regression problem
%  DATA = LOAD_DATA(DATA) preprocess data for training
%  Read in data as a table. Return data as tables.
%
%  Example:
%  x,y = importfile(data_table);
%
  data(any(ismissing(data),2),:) = [];

  x = data(:,{'x','x1','x2','x3','x4'});  
  y = data(:,{'stop'});
end