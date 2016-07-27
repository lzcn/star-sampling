%% maximum budget
out_dir = './BUDGETMAXIMUM';
if(~isdir(out_dir))
    mkdir(out_dir);
end
samples = power(10,3:7);
budget = power(10,3:7);
TRIAL001(data_path,out_dir,samples,budget,1);
%% 1k budget
out_dir = './BUDGET1K';
if(~isdir(out_dir))
    mkdir(out_dir);
end
samples = power(10,3:7);
budget = 1e3*ones(size(samples));
TRIAL001(data_path,out_dir,samples,budget,1);

%% 10k budget
out_dir = './BUDGET10K';
if(~isdir(out_dir))
    mkdir(out_dir);
end
samples = power(10,4:7);
budget = 1e4*ones(size(samples));
TRIAL001(data_path,out_dir,samples,budget,1);