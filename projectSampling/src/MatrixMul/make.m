mex ('-v','-largeArrayDims',...
    'diamondMatrix.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath], ...
    '-outdir',bin_path);
mex ('-v','-largeArrayDims',...
    'wedgeMatrix.cpp',...
    [src_path,'matrix.cpp'],['-I',ipath], ...
    '-outdir',bin_path);
