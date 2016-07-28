out_dir = 'result';
if(~idsir(out_dir))
    mkdir(out_dir); 
end
samples = power(10,3:6);
budget = 1e3;
knn = 100;
turn = 10;
TRIAL002(data_path, out_dir, samples, budget, knn, turn);
