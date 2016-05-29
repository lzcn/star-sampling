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

% make all
cd src/ExactSearch/;
make;
cd ../MatrixMul/;
make;
cd ../../;
% add path
addpath('bin');