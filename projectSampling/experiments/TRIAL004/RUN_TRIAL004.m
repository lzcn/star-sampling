mex('-largeArrayDims',...
    'diamondExp004.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath]);
mex('-largeArrayDims',...
    'equalityExp004.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath]);

% range of budget
budget = power(10,2:7);
% the top_t
top_t = budget(1);
% number of samples
samples = budget(end);
% times of experiments
turn = 10;
% folder to save result
out_dir = './result';
if(~isdir(out_dir))
    mkdir(out_dir);
end

ex004(data_path,out_dir,samples,budget,top_t,turn);