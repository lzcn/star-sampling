addpath('bin'); clear all; clc;

%% experiments for data "lastfm"

% load data
load('data/lastfm/A.mat');
load('data/lastfm/B.mat');
load('data/lastfm/C.mat');
load('data/lastfm/valueQuery.mat');
load('data/lastfm/timeQuery.mat');
load('data/lastfm/exactTime.mat');
load('data/lastfm/valueTrue.mat');
% choice
wedge = false;
diamond = true;
equality = true;
triwedge = false;
query = false;
% vars to record
samples = power(10,4:7); % number of samples
budget = 10e4*ones(size(samples));
top_t = power(10,0:3); % find the top-t value
varSize = [size(samples,2),size(top_t,2)];
% recall(s,t): the recall of top_t under the sample number of s
diamondRecall = zeros(varSize); % recall of diamond sampling 
wedgeRecall = zeros(varSize); % recall of wedge sampling
equalityRecall = zeros(varSize);
triwedgeRecall = zeros(varSize);
% time consuming
diamondTimes = zeros(size(samples));
wedgeTimes = zeros(size(samples));
equalityTimes = zeros(size(samples));
triwedgeTimes = zeros(size(samples));
% sampling

for i = 1:length(samples)
    for j = 1 : length(top_t)
        t = top_t(j);
        % wedge sampling
        if wedge
        [wedgeValues, wedgeTimes(i), ~] = wedgeTensor(A', B, C, ...
                                    budget(i),samples(i),top_t(j));
        wedgeRecall(i,j) = sum(wedgeValues(1:t) >= valueTrue(t))/t;
        end
        % diamond sampling
        if diamond
        [diamondValues, diamondTimes(i), ~] = diamondTensor(A', B, C, ...
                                    budget(i),samples(i),top_t(j));
        diamondRecall(i,j) = sum(diamondValues(1:t) >= valueTrue(t))/t;
        end
        % equality sampling
        if equality
        [euqalityValues, equalityTimes(i), ~] = equalitySampling(A, B, C, ...
                                    budget(i),samples(i),top_t(j));
        equalityRecall(i,j) = sum(euqalityValues(1:t) >= valueTrue(t))/t;
        end
        % tri-wedge Sampling
        if triwedge
        [triwedgeValues, triwedgeTimes(i), ~] = triWedgeSampling(A', B, C, ...
                                    budget(i),samples(i),top_t(j));
        triwedgeRecall(i,j) = sum(triwedgeValues(1:t) >= valueTrue(t))/t;
        end
    end
end
if false
% draw time - sample
timeSample = figure; hold on;title('Time-Samples'); 
xlabel('log_{10}Samples'); 
ylabel('log_{10}T(sec)'); 
plot(log10(samples),log10(exactTime),'b','LineWidth',2);
plot(log10(samples),log10(diamondTimes),'r','LineWidth',2); 
plot(log10(samples),log10(wedgeTimes),'--g','LineWidth',2);
legend('exhaustive','diamond','wedge');
% saveas(timeSample,'sample-time.png'); 
% draw recall - sample
recallSample = figure; hold on; title('Recall'); 
xlabel('log_{10}Samples'); 
ylabel('recall');
c = ['r','b','k','g'];
for i = 1:size(top_t,2)
    plot(log10(samples),diamondRecall(:,i),c(i),'LineWidth',2);
    plot(log10(samples),wedgeRecall(:,i),['--',c(i)],'LineWidth',2); 
end
legend('diamond:t=1','wedge:t=1',...
       'diamond:t=10','wedge:t=10',...
       'diamond:t=100','wedge:t=100',...
       'diamond:t=1000','wedge:t=1000');  
saveas(recallSample,'sample-recall.png'); 

end

% diamond sampling for query
if query
% query sampling
KNN = 100;
numQueries = 1000;
recall = zeros(numQueries,2);
nonZero = sum(abs(A(:,1:numQueries)))~=0;
timeQuery = timeQuery/1000;
for i = 1:length(samples)
    [sValue,sTime] = querySampling(A(:,1:numQueries),B,C,samples(i),samples(i),KNN);
    for j = 1: numQueries
        if(nonZero(j) == 1)
            [~,dValue] = diamondTensor(A(:,j),B,C,samples(i),samples(i));
            recall(j,1) = sum(sValue(1:KNN,j) >= valueQuery(KNN,j))/KNN;
            recall(j,2) = sum(dValue(1:KNN)>= valueQuery(KNN,j))/KNN;
        end
    end
    recalla = recall(nonZero,1);
    recallb = recall(nonZero,2);
    filename = ['Recall-Equeries with 10^',num2str(log10(samples(i))),' samples'];
    h = figure;hold on; title(filename);
    xlabel('qeuaries'); 
    ylabel('recall');
    axis([1 1000 0 1]);
    plot(1:10:size(recalla,1),recalla(1:10:size(recalla,1)),'--r');
    plot(1:10:size(recalla,1),recallb(1:10:size(recallb,1)),'b');
    legend('query','General');
    saveas(h,[filename,'.png']); 
    t = sTime(nonZero);
    filename = ['Time-Equeries with 10^',num2str(log10(samples(i))),' samples'];
    h = figure;hold on; title(filename);
    xlabel('qeuaries'); 
    ylabel('time');
    plot(1:10:size(t,1),t(1:10:size(recalla,1)),'r');
    saveas(h,[filename,'.png']); 
end
end

