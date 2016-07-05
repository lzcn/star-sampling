clc;clear all;
addpath('bin');
% load data
load('data\lastfm\A.mat');
load('data\lastfm\B.mat');
load('data\lastfm\C.mat');
KNN = 1000;
[result,time] = queryFullSearch(A(1:1000,:)',B',C',KNN);
% save result
save('data/lastfm/valueQuery.mat','result');
save('data/lastfm/timeQuery.mat','time');