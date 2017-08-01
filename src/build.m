mex ('-v','-largeArrayDims',...
    'factorization.cpp',['-I',ipath], ...
    '-outdir',bin_path);
mex ('-v','-largeArrayDims',...
    'trainFashion.cpp',['-I',ipath], ...
    '-outdir',bin_path);
mex ('-v','-largeArrayDims',...
    'testrandom.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath], ...
    '-outdir',bin_path);
% build BPTF
if(isdir('BPTF'))
    cd ./BPTF
    % copy functions into lib
    copyfile('BPMF.m',lib_path);
    copyfile('BPMF_Predict.m',lib_path);
    copyfile('BPTF.m',lib_path);
    copyfile('BPTF_Predict.m',lib_path);
    copyfile('PMF_Grad.m',lib_path);
    mex ('-v','-largeArrayDims','PMF_Reconstruct.cpp','-outdir',bin_path);
    mex ('-v','-largeArrayDims','PMF_Grad_Unit.cpp','-outdir',bin_path);
    mex ('-v','-largeArrayDims','PTF_Reconstruct.cpp','-outdir',bin_path);
    mex ('-v','-largeArrayDims','PTF_ComputeQ.cpp','-outdir',bin_path);
    % copy lib files into pathto/lib
    copyfile('lib/cell2vars.m',lib_path);
    copyfile('lib/EncodeInt.m',lib_path);
    copyfile('lib/GetOptions.m',lib_path);
    copyfile('lib/GroupIndex.m',lib_path);
    copyfile('lib/mvnrndpre.m',lib_path);
    copyfile('lib/RI.m',lib_path);
    copyfile('lib/RMSE.m',lib_path);
    % copy class into pathto/lib/@spTensor
    copyfile('@spTensor',fullfile(lib_path,'@spTensor'));
    mex ('-v','-largeArrayDims','./lib/EncodeInt32Array.cpp','-outdir',lib_path);
    mex ('-v','-largeArrayDims','./lib/EncodeInt64Array.cpp','-outdir',lib_path);
    cd ../;
else
   disp('No Libary for BPTF!');
end
