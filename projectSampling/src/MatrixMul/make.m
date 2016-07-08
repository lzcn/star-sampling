% include path
ipath = ['-I' '../../include'];
src_path = '../';
mex ('-v','-largeArrayDims',...
    'diamondMatrix.cpp',...
    [src_path,'matrix.cpp'],ipath, ...
    '-outdir','../../bin');
mex ('-v','-largeArrayDims',...
    'wedgeMatrix.cpp',...
    [src_path,'matrix.cpp'],ipath, ...
    '-outdir','../../bin');