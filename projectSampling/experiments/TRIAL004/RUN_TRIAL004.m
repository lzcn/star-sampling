mex('-largeArrayDims',...
    'diamondTRIAL004.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath]);
mex('-largeArrayDims',...
    'equalityTRIAL004.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath]);
mex('-largeArrayDims',...
    'extensionTRIAL004.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath]);
% range of budget
budget = power(10,2:7);
% the top_t
top_t = budget(1);
% number of samples
samples = budget(end);
% times of experiments
turn = 1;
% folder to save result
out_dir = './result';
if(~isdir(out_dir))
    mkdir(out_dir);
end

TRIAL004(data_path,out_dir,samples,budget,top_t,turn);