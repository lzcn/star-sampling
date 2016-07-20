function ex001(data_path, out_dir, samples, budget,turn)

    ml_10m_path = fullfile(data_path,'MovieLens','ml-10m');
    ml_20m_path = fullfile(data_path,'MovieLens','ml-20m');
    ml_2k_path = fullfile(data_path, 'hetrec2011-movielens-2k-v2');
    lastfm_path = fullfile(data_path, 'hetrec2011-lastfm-2k');
    delicious_path = fullfile(data_path, 'hetrec2011-delicious-2k');
    
    top_t = power(10,0:3); 
    varSize = [size(samples,2),size(top_t,2)];
    %% ml-10m
     dataName = 'ml-10m';
 
     load(fullfile(ml_10m_path,'User.Mat'));
     load(fullfile(ml_10m_path,'Movie.Mat'));
     load(fullfile(ml_10m_path,'Tag.Mat'));
     load(fullfile(ml_10m_path,'topValue.Mat'));
     load(fullfile(ml_10m_path,'fullTime.Mat'));
     load(fullfile(ml_10m_path,'topIndexes.Mat'));
 
     oneDataSet(dataName, out_dir, ...
                     varSize, samples, budget, top_t, ...
                     User, Movie, Tag,...
                     fullTime,topValue,topIndexes,turn);
     %%
     dataName = 'ml-20m';
 
     load(fullfile(ml_20m_path,'User.Mat'));
     load(fullfile(ml_20m_path,'Movie.Mat'));
     load(fullfile(ml_20m_path,'Tag.Mat'));
     load(fullfile(ml_20m_path,'topValue.Mat'));
     load(fullfile(ml_20m_path,'fullTime.Mat'));
     load(fullfile(ml_20m_path,'topIndexes.Mat'));
 
     oneDataSet(dataName, out_dir, ...
                     varSize, samples, budget, top_t, ...
                     User, Movie, Tag,...
                     fullTime,topValue,topIndexes,turn);
     %%
     dataName = 'ml-2k';
 
     load(fullfile(ml_2k_path,'User.Mat'));
     load(fullfile(ml_2k_path,'Movie.Mat'));
     load(fullfile(ml_2k_path,'Tag.Mat'));
     load(fullfile(ml_2k_path,'topValue.Mat'));
     load(fullfile(ml_2k_path,'fullTime.Mat'));
     load(fullfile(ml_2k_path,'topIndexes.Mat'));
 
     oneDataSet(dataName, out_dir, ...
                     varSize, samples, budget, top_t, ...
                     User, Movie, Tag,...
                     fullTime,topValue,topIndexes,turn);    
     %%
 
     dataName = 'lastfm';
 
     load(fullfile(lastfm_path,'User.Mat'));
     load(fullfile(lastfm_path,'Artist.Mat'));
     load(fullfile(lastfm_path,'Tag.Mat'));
     load(fullfile(lastfm_path,'topValue.Mat'));
     load(fullfile(lastfm_path,'fullTime.Mat'));
     load(fullfile(lastfm_path,'topIndexes.Mat'));
 
     oneDataSet(dataName, out_dir, ...
                     varSize, samples, budget, top_t, ...
                     User, Artist, Tag,...
                     fullTime,topValue,topIndexes,turn);      
     %%
     dataName = 'delicious';
 
     load(fullfile(delicious_path,'User.Mat'));
     load(fullfile(delicious_path,'Url.Mat'));
     load(fullfile(delicious_path,'Tag.Mat'));
     load(fullfile(delicious_path,'topValue.Mat'));
     load(fullfile(delicious_path,'fullTime.Mat'));
     load(fullfile(delicious_path,'topIndexes.Mat'));
 
     oneDataSet(dataName, out_dir, ...
                     varSize, samples, budget, top_t, ...
                     User, Url, Tag,...
                     fullTime,topValue,topIndexes,turn);
    %% random data
   dataName = 'random';
   load('A.Mat');
   load('B.Mat');
   load('C.Mat');
   topValue = 1600*ones(1000,1);
   topIndexes = ones(1000,3);
   count = 1;
   for i = 1:10
       for j = 1:10
           for k = 1:10
               topIndexes(count,1) = i;
               topIndexes(count,2) = j;
               topIndexes(count,3) = k;
           end
       end
   end
   fullTime = 1e4;
   oneDataSet(dataName, out_dir, ...
                   varSize, samples, budget, top_t, ...
                   A, B, C,...
                   fullTime,topValue,topIndexes,turn);    
end

function [ diamondRecall, diamondTimes, ...
           equalityRecall, equalityTimes ] = initVar(varSize, samples)

    diamondRecall = zeros(varSize);
    equalityRecall = zeros(varSize);
    diamondTimes = zeros(size(samples));
    equalityTimes = zeros(size(samples));

end


function [ diamondRecall, diamondTimes, ...
           equalityRecall, equalityTimes ] = oneSampling(varSize, samples, budget, top_t, ...
                                                            A, B, C,...
                                                            topValue,turn)
    [ diamondRecall, diamondTimes, ...
           equalityRecall, equalityTimes ] = initVar(varSize, samples);
    for s = 1:turn
        diamondRecalltemp = zeros(size(diamondRecall));
        diamondTimestemp = zeros(size(diamondTimes));
        equalityRecalltemp = zeros(size(equalityRecall));
        equalityTimestemp = zeros(size(equalityTimes));
        for i = 1:length(samples)
            for j = 1 : length(top_t)
                t = top_t(j);
        
                [diamondValues, diamondTimestemp(i), ~] = diamondTensor(A', B, C, ...
                                            budget(i),samples(i),top_t(j));
                diamondRecalltemp(i,j) = sum(diamondValues(1:t) >= topValue(t))/t;
        
                [euqalityValues, equalityTimestemp(i), ~] = equalitySampling(A, B, C, ...
                                            budget(i),samples(i),top_t(j));
                equalityRecalltemp(i,j) = sum(euqalityValues(1:t) >= topValue(t))/t;
            end
        end
        diamondRecall = diamondRecall + diamondRecalltemp;
        diamondTimes = diamondTimes + diamondTimestemp;
        equalityRecall = equalityRecall + equalityRecalltemp;
        equalityTimes = equalityTimes + equalityTimestemp;
    end
    diamondRecall = diamondRecall/turn;
    diamondTimes = diamondTimes/turn;
    equalityRecall = equalityRecall/turn;
    equalityTimes = equalityTimes/turn;
end

function drawTimeFig(titlename, out_dir, samples, fullTime, diamondTimes, equalityTimes)
    h = figure; hold on;title(titlename); 
    xlabel('log_{10}Samples'); 
    ylabel('log_{10}T(sec)'); 
    plot(log10(samples),log10(fullTime*ones(size(samples))),'b','LineWidth',2);
    plot(log10(samples),log10(diamondTimes),'r','LineWidth',2); 
    plot(log10(samples),log10(equalityTimes),'--g','LineWidth',2);
    legend('exhaustive','diamond','equality',4);
    saveas(h,fullfile(out_dir,[titlename,'.png']));
end

function drawRecallFig(titlename, out_dir, samples,top_t, diamondRecall, equalityRecall)
    h = figure; hold on; title(titlename); 
    xlabel('log_{10}Samples'); 
    ylabel('recall');
    axis([log10(samples(1)) log10(samples(end)) 0 1.1]);
    c = ['r','b','k','g', 'c'];
    desc = {};
    for i = 1:size(top_t,2)
        plot(log10(samples),diamondRecall(:,i),c(i),'LineWidth',2);
        desc{end+1} = ['d:t=',num2str(top_t(i))];
        plot(log10(samples),equalityRecall(:,i),['--',c(i)],'LineWidth',2); 
        desc{end+1} = ['e:t=',num2str(top_t(i))];
    end
    legend(desc,4);  
    saveas(h,fullfile(out_dir,[titlename,'.png'])); 

end

function oneDataSet(dataName, out_dir, ...
                    varSize, samples, budget, top_t, ...
                    A, B, C,...
                    fullTime,topValue,topIndexes,turn)
    [ diamondRecall, diamondTimes, ...
           equalityRecall, equalityTimes ] = oneSampling(varSize, samples, budget, top_t, ...
                                                            A, B, C,...
                                                            topValue,turn);
    titlename = ['time-samples-',dataName];
    drawTimeFig(titlename, out_dir, samples, fullTime, diamondTimes, equalityTimes);
    titlename = ['recall-samples-',dataName];
    drawRecallFig(titlename, out_dir, samples, top_t, diamondRecall, equalityRecall);
    [p1,p2] = getProbability(A,B,C,topIndexes,topValue);
    titlename = ['probability-instances-',dataName];
    h = figure; hold on; title(titlename);plot(p1,'b');plot(p2,'r');legend('diamond','equality');
    saveas(h,fullfile(out_dir,[titlename,'.png']));
end

function [p1,p2] = getProbability(A, B, C, topIndexes,topValue)
    weightD = zeros(size(A));
    weightR = zeros(size(A,2),1);
    for r = 1 : size(A,2)
        weightR(r) = sum(abs(A(:,r)))*sum(B(:,r))*sum(C(:,r));
        for i = 1 : size(A,1)
            weightD(i,r) = abs(A(i,r))*sum(abs(A(i,:)))*sum(B(:,r))*sum(C(:,r));
        end
    end
    sum1 = sum(sum(weightD));
    sum2 = sum(sum(weightR));
    p1 = zeros(size(topValue));
    p2 = zeros(size(topValue));
    for t = 1: length(topValue)
        idx = topIndexes(t,1);
        p1(t) = topValue(t)*sum(abs(A(idx,:)))/sum1;
        p2(t) = topValue(t)/sum2;
    end
end