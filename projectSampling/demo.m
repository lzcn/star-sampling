clc;
addpath('bin');
cd data\as-skitter\
A = getData();
cd ..\..\
diamond_sparse(A',A,100,1e2);