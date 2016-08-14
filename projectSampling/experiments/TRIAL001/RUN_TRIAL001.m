paths =   { '../../data/hetrec2011-delicious-2k', ...
            '../../data/hetrec2011-lastfm-2k', ...
            '../../data/hetrec2011-movielens-2k-v2', ...
            '../../data/MovieLens/ml-10m', ...
            '../../data/MovieLens/ml-20m'};
            
dataName ={ 'delicious',...
            'lastfm', ...
            'ml-2k', ...
            'ml-10m', ...
            'ml-20m'};

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
TRIAL001(paths,dataName,out_dir,budget,samples,top_t,turn,true);
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
TRIAL001(paths,dataName,out_dir,budget,samples,top_t,turn,true);
