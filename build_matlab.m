function output = build_matlab(action)
path = pwd;
src_path = fullfile(path, 'src');
bin_path = fullfile(path, 'bin');
build_path = fullfile(path, 'build');
inc_path = fullfile(path, 'include');
IFLAG = ['-I', inc_path];
if nargin > 0 && action == "clean"
    delete("build/*.o");
    delete("bin/*.mex*");
    disp("Done clean.");
elseif nargin > 0 && action == "mex"
    % cpp files for mex
    cpp_files = dir('mex/**/*.cpp');
    % object files from source
    obj_files = fullfile('build', {dir('build/*.o').name});
    % build mex files
    for i = 1:length(cpp_files)
        src_file = fullfile(cpp_files(i).folder, cpp_files(i).name);
        mex('-v', '-O', '-largeArrayDims', IFLAG, '-outdir', bin_path, src_file, obj_files{:});
    end
    addpath(bin_path);
    disp("Done build.")
elseif nargin > 0 && action == "obj"
    %% build objective files
    src_files = dir('src/**/*.cpp');
    for i = 1:length(src_files)
        src_file = fullfile(src_files(i).folder, src_files(i).name);
        mex('-c',  IFLAG, '-outdir', build_path, src_file);
    end
else
    %% build objective files
    src_files = dir('src/**/*.cpp');
    for i = 1:length(src_files)
        src_file = fullfile(src_files(i).folder, src_files(i).name);
        mex('-c', IFLAG, '-outdir', build_path, src_file);
    end
    %% build mex files
    % cpp files for mex
    cpp_files = dir('mex/**/*.cpp');
    % object files from source
    obj_files = fullfile('build', {dir('build/*.o').name});
    % build mex files
    for i = 1:length(cpp_files)
        src_file = fullfile(cpp_files(i).folder, cpp_files(i).name);
        mex('-v', '-O', '-largeArrayDims', IFLAG, '-outdir', bin_path, src_file, obj_files{:});
    end
    addpath(bin_path);
    disp("Done build.")
end
end