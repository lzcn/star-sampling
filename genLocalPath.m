path = pwd;
ipath = fullfile(path,'include');
src_path = fullfile(path,'src/');
bin_path = fullfile(path,'bin');
lib_path = fullfile(path,'lib');
data_path = fullfile(path,'data');
addpath(bin_path);
addpath(genpath(lib_path));
