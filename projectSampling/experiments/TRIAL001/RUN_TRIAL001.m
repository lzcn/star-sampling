%% maximum budget
out_dir = './max';
if(~isdir(out_dir))
    mkdir(out_dir);
end
samples = power(10,3:7);
top_t = power(10,0:3);
budget = zeros(length(samples),length(top_t));
for t = 1:length(top_t)
    budget(:,t) = power(10,3:7);
end
turn = 20;
TRIAL001(data_path,out_dir,budget,samples,top_t,turn,false);
%% 1k budget
out_dir = './budget';
if(~isdir(out_dir))
    mkdir(out_dir);
end
samples = power(10,4:7);
top_t = power(10,0:3);
budget = zeros(length(samples),length(top_t));
for t = 1:length(top_t)
    budget(:,t) = 10*top_t(t);
end
turn = 20;
TRIAL001(data_path,out_dir,budget,samples,top_t,turn,false);