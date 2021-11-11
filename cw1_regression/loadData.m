function data = load_data(filename, dataLines)
%LOAD_DATA LOAD data from a text file
%  DATA = LOAD_DATA(FILENAME) reads data from text file
%  FILENAME for the default selection.  Returns the data as a table.
%
%  DATA = IMPORTFILE(FILE, DATALINES) reads data for the
%  specified row interval(s) of text file FILENAME. Specify DATALINES as
%  a positive scalar integer or a N-by-2 array of positive scalar
%  integers for dis-contiguous row intervals.
%
%  Example:
%  E50C3000N5Cov = importfile("data/train.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 01-Nov-2021 17:22:10

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["nid", "status", "start", "stop", "z", "x", "x1", "x2", "x3", "x4"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
data = readtable(filename, opts);

end