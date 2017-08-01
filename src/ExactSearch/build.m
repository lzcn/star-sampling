% mex
cpp_files = dir('./*.cpp');
for i = 1:length(cpp_files)
    mex ('-v','-largeArrayDims',cpp_files(i).name,[src_path,'matrix.cpp'],['-I',ipath],'-outdir',bin_path);
end