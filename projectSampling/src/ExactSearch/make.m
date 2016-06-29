% include path
ipath = ['-I' '../../include'];


% src path
src_path = '../';
mex ('-v','-largeArrayDims',...
    'exact_search_three_order_tensor.cpp',...
    [src_path,'matrix.cpp'],ipath, ...
    '-outdir','../../bin');
mex ('-v','-largeArrayDims',...
    'exact_search.cpp',...
    [src_path,'matrix.cpp'],ipath, ...
    '-outdir','../../bin');