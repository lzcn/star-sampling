% matlab makefile
mex -setup C++
% include file path
ipath = ['-I' 'include'];
% source file path
src_path = 'src\';
% blaslib path
blaslib = fullfile(matlabroot,'extern','lib',computer('arch'),...
    'microsoft','libmwblas.lib');
%lapacklib path
lapacklib = fullfile(matlabroot,'extern','lib',computer('arch'),...
    'microsoft','libmwlapack.lib');

% compile exact_search.cpp
mex ('-v','-largeArrayDims',[src_path,'exact_search.cpp']);
% compile merge.cpp
mex ('-v',[src_path,'merge.cpp']);
% compile diamondsamping
mex ('-v','-largeArrayDims',[src_path,'diamondsampling.cpp'],[src_path,'tensor.cpp'],ipath);