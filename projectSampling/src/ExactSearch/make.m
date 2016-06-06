% include path
ipath = ['-I' '../../include'];
% bin path
outdir = ['-outdir' '../../bin'];
% src path
src_path = ['../'];
mex ('-v','-largeArrayDims',...
    'diamondsampling.cpp',[src_path,'tensor.cpp'],ipath,outdir);