clc;
clear all;
mex -g diamond_matrix.cpp;
A = [1,2,5;3,4,6];
B = [1,1;1,1];

diamond_matrix(A,B,1,1000);