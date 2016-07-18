out_dir = './maximumbudget';
samples = power(10,3:7);
budget = power(10,3:7);
tic
ex001(data_path,out_dir,samples,budget);
toc