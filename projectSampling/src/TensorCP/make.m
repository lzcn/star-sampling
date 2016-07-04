% include path
ipath = ['-I' '../../include'];


% src path
src_path = '../';
mex ('-v','-largeArrayDims',...
    'diamondTensor.cpp',...
    [src_path,'matrix.cpp'],ipath, ...
    '-outdir','../../bin');
mex ('-v','-largeArrayDims',...
    'wedgeTensor.cpp',...
    [src_path,'matrix.cpp'],ipath, ...
    '-outdir','../../bin');

mex ('-v','-largeArrayDims',...
    'querySampling.cpp',...
    [src_path,'matrix.cpp'],ipath, ...
    '-outdir','../../bin');