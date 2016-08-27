path =  '../../data/hetrec2011-movielens-2k-v2';
%% maximum budget
out_dir = './max/ml-2k';
if(~isdir(out_dir))
    mkdir(out_dir);
end
samples = power(10,9);
budget = samples;
top_t = power(10,0:3);
turn = 10;

fullTime = 0;
topValue = [];
load(fullfile(path,'User.mat'));
load(fullfile(path,'Item.mat'));
load(fullfile(path,'Tag.mat'));
load(fullfile(path,'topValue.mat'));
load(fullfile(path,'fullTime.mat'));
User = User'; Item = Item'; Tag = Tag';

varSize = [length(samples),length(top_t)];
CRecall = zeros(varSize);
ERecall = zeros(varSize);
CTimes  = zeros(length(samples),1);
ETimes  = zeros(length(samples),1);

for TurnNum = 1:turn
    CTempRecall = zeros(varSize);
    ETempRecall = zeros(varSize);
    CTempTimes  = zeros(length(samples),1);
    ETempTimes  = zeros(length(samples),1);
    for i = 1 : length(samples)
        s = samples(i);
        tp = budget(i);
        [cValue, cTime, ~] = centralSampling(User,Item,Tag,tp,s,1000);
        [eValue, eTime, ~] = extensionSampling(User,Item,Tag,tp,s,1000);
        for j = 1 : length(top_t)
            t = top_t(j);
            truevalue = topValue(t);
            CTempRecall(i,j) = sum(cValue(1:t) >= truevalue)/t;
            ETempRecall(i,j) = sum(eValue(1:t) >= truevalue)/t;
        end
        CTempTimes(i) = cTime;
        ETempTimes(i) = eTime;
    end
    CRecall = CRecall + CTempRecall;
    ERecall = ERecall + ETempRecall;
    CTimes = CTimes + CTempTimes;
    ETimes = ETimes + ETempTimes;
end
CRecall= CRecall/turn;
CTimes = CTimes /turn;
% save(fullfile(out_dir,'CRecall.mat'),'CRecall');
% save(fullfile(out_dir,'CTimes.mat'),'CTimes');
ERecall= ERecall/turn;
ETimes = ETimes /turn;
% save(fullfile(out_dir,'ERecall.mat'),'ERecall');
% save(fullfile(out_dir,'ETimes.mat'),'ETimes');
