function output = build_octave(action)
path = pwd;
src_path = fullfile(path, 'src');
bin_path = fullfile(path, 'bin');
build_path = fullfile(path, 'build');
mkdir("bin");
mkdir("build");
if nargin > 0 && strcmp(action, "clean")
    delete("bin/*.mex")
    delete("build/*.o")
    disp("Done clean.");
elseif nargin > 0 && strcmp(action, "mex")
    %% build mex files
    % cpp files for mex
    cpp_files = glob('mex/*/*.cpp');
    % object files from source
    obj_files = fullfile('build', {dir('build/*.o').name});
    % build mex files
    for i = 1:length(cpp_files)
        src_file = cpp_files{i};
        [folder, name, ext] = fileparts(src_file);
        out_file = fullfile(bin_path, [name, ".mex"]);
        mex('-o', out_file, '--std=c++11', src_file);
    end
    addpath(bin_path);
    disp("Done build.")
elseif nargin > 0 && strcmp(action, "obj")
    %% build objective files
    src_files = dir('src/*.cpp');
    for i = 1:length(src_files)
        src_name = src_files(i).name;
        src_file = fullfile(src_files(i).folder, src_name);
        out_file = fullfile(build_path, strrep(src_name, ".cpp", ".o"))
        mex('-c', '-o', out_file, '--std=c++11', src_file);
    end
else
    %% build objective files
    src_files = dir('src/*.cpp');
    for i = 1:length(src_files)
        src_name = src_files(i).name;
        src_file = fullfile(src_files(i).folder, src_name);
        out_file = fullfile(build_path, strrep(src_name, ".cpp", ".o"))
        mex('-c', '-o', out_file, '--std=c++11', src_file);
    end
    %% build mex files
    % cpp files for mex
    cpp_files = glob('mex/*/*.cpp');
    % object files from source
    obj_files = fullfile('build', {dir('build/*.o').name});
    % build mex files
    for i = 1:length(cpp_files)
        src_file = cpp_files{i};
        [folder, name, ext] = fileparts(src_file);
        out_file = fullfile(bin_path, [name, ".mex"]);
        mex('-v', '-o', out_file, '--std=c++11', src_file);
    end
    addpath(bin_path);
    disp("Done build.")
end
end