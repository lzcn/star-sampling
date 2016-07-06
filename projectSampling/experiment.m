addpath('bin'); clc;

%% lastfm
% load data
load('data/lastfm/A.mat');
load('data/lastfm/B.mat');
load('data/lastfm/C.mat');
load('data/lastfm/valueQuery.mat');
load('data/lastfm/timeQuery.mat');
load('data/lastfm/exactTime.mat');
load('data/lastfm/valueTrue.mat');

% vars to record
samples = power(10,3:7); % number of samples
top = power(10,0:3); % find the top-t value
varSize = [size(samples,2),size(top,2)];
% recall(s,t): the recall of topt under the sample number of s
diamondRecall = zeros(varSize); % recall of diamond sampling 
wedgeRecall = zeros(varSize); % recall of wedge sampling
diamondTimes = zeros(size(samples));
wedgeTimes = zeros(size(samples));
exactTime = exactTime*ones(size(samples));
A = A';
% sampling
for i = 1:size(samples,2)
    % wedge sampling
    tic;
    [~,wedgeValues] = wedgeTensor(A,B,C,samples(i),samples(i)); 
    wedgeTimes(i) = toc;
    % diamond sampling
    tic; 
    [~,diamondValues] = diamondTensor(A,B,C,samples(i),samples(i));
    diamondTimes(i) = toc;
    % find the recall of topt
    for j = 1 : size(top,2)
        t = top(j);
        if(size(diamondValues,1) < t)
            t = size(diamondValues,1);
        end
        diamondRecall(i,j) = sum(diamondValues(1:t) >= valueTrue(t))/t;
        t = top(j);
        if(size(wedgeValues,1) < t)
            t = size(wedgeValues,1);
        end        
        wedgeRecall(i,j) = sum(wedgeValues(1:t) >= valueTrue(t))/t; 
    end
end

% draw time - sample
timeSample = figure; hold on;title('Time-Samples'); 
xlabel('log_{10}Samples'); 
ylabel('log_{10}T(sec)'); 
plot(log10(samples),log10(exactTime),'b','LineWidth',2);
plot(log10(samples),log10(diamondTimes),'r','LineWidth',2); 
plot(log10(samples),log10(wedgeTimes),'--g','LineWidth',2);
legend('exhaustive','diamond','wedge');
saveas(timeSample,'sample-time.png'); 
% draw recall - sample
recallSample = figure; hold on; title('Recall'); 
xlabel('log_{10}Samples'); 
ylabel('recall');
c = ['r','b','k','g'];

for i = 1:size(top,2)
    plot(log10(samples),diamondRecall(:,i),c(i),'LineWidth',2);
    plot(log10(samples),wedgeRecall(:,i),['--',c(i)],'LineWidth',2); 
end
legend('diamond:t=1','wedge:t=1',...
       'diamond:t=10','wedge:t=10',...
       'diamond:t=100','wedge:t=100',...
       'diamond:t=1000','wedge:t=1000');  
saveas(recallSample,'sample-recall.png'); 


% query sampling
KNN = 100;
numQueries = 1000;
recall = zeros(numQueries,2);
nonZero = sum(abs(A(:,1:numQueries)))~=0;
% diamond sampling for query
for i = 1:size(samples,2)
    [sValue,sTime] = querySampling(A(:,1:numQueries),B,C,samples(i),samples(i),KNN);
    parfor j = 1: numQueries
        if(nonZero(j) == 1)
            [~,dValue] = diamondTensor(A(:,j),B,C,samples(i),samples(i));
            recall(j,:) = ...
                [sum(sValue(1:KNN,j) >= valueQuery(KNN,j))/KNN,...
                 sum(dValue(1:KNN)>= valueQuery(KNN,j))/KNN];
        end
    end
end

