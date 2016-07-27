function TRIAL002(data_path, out_dir, samples, budget, knn, turn)

     ml_10m_path = fullfile(data_path,'MovieLens','ml-10m');
     ml_20m_path = fullfile(data_path,'MovieLens','ml-20m');
     ml_2k_path = fullfile(data_path, 'hetrec2011-movielens-2k-v2');
     lastfm_path = fullfile(data_path, 'hetrec2011-lastfm-2k');
     delicious_path = fullfile(data_path, 'hetrec2011-delicious-2k');

     %% ml-10m
     dataName = 'ml-10m';

     load(fullfile(ml_10m_path,'User.Mat'));
     load(fullfile(ml_10m_path,'Movie.Mat'));
     load(fullfile(ml_10m_path,'Tag.Mat'));
     load(fullfile(ml_10m_path,'valueQuery.Mat'));
     load(fullfile(ml_10m_path,'timeQuery.Mat'));

     oneDataSet(dataName, out_dir, ...
                samples, budget, knn, ...
                User, Movie, Tag,...
                timeQuery,valueQuery,...
                turn);
     %% ml-20m
     dataName = 'ml-20m';
 
     load(fullfile(ml_20m_path,'User.Mat'));
     load(fullfile(ml_20m_path,'Movie.Mat'));
     load(fullfile(ml_20m_path,'Tag.Mat'));
     load(fullfile(ml_20m_path,'valueQuery.Mat'));
     load(fullfile(ml_20m_path,'timeQuery.Mat'));
 
     oneDataSet(dataName, out_dir, ...
                samples, budget, knn, ...
                User, Movie, Tag,...
                timeQuery,valueQuery,...
                turn);
     %% ml-2k
     dataName = 'ml-2k';
 
     load(fullfile(ml_2k_path,'User.Mat'));
     load(fullfile(ml_2k_path,'Movie.Mat'));
     load(fullfile(ml_2k_path,'Tag.Mat'));
     load(fullfile(ml_2k_path,'valueQuery.Mat'));
     load(fullfile(ml_2k_path,'timeQuery.Mat'));
 
     oneDataSet(dataName, out_dir, ...
                samples, budget, knn, ...
                User, Movie, Tag,...
                timeQuery,valueQuery,...
                turn);   
     %% lastfm
 
     dataName = 'lastfm';
 
     load(fullfile(lastfm_path,'User.Mat'));
     load(fullfile(lastfm_path,'Artist.Mat'));
     load(fullfile(lastfm_path,'Tag.Mat'));
     load(fullfile(lastfm_path,'valueQuery.Mat'));
     load(fullfile(lastfm_path,'timeQuery.Mat'));

     oneDataSet(dataName, out_dir, ...
                samples, budget, knn, ...
                User, Artist, Tag,...
                timeQuery,valueQuery,...
                turn);     
      
     %% delicious
     dataName = 'delicious';
 
     load(fullfile(delicious_path,'User.Mat'));
     load(fullfile(delicious_path,'Url.Mat'));
     load(fullfile(delicious_path,'Tag.Mat'));
     load(fullfile(delicious_path,'valueQuery.Mat'));
     load(fullfile(delicious_path,'timeQuery.Mat'));
     
     oneDataSet(dataName, out_dir, ...
                samples, budget, knn, ...
                User, Url, Tag,...
                timeQuery,valueQuery,...
                turn);
end

%% Initialize Variables
function [ diamond, equality, extension ] = initVar(Variables)
    varSize = [length(Variables.samples), Variables.NumQueries];
    diaond.recall = zeros(varSize);
    diaond.time = zeros(varSize);
    equality.recall = zeros(varSize);
    equality.time = zeros(varSize);
    extension.recall = zeros(varSize);
    extension.time = zeros(varSize);
end
function [ diamondRecall, diamondTimes, equalityRecall, equalityTimes ] = initVar(varSize)
    % recall(i,j) is the recall of j-th query under sampling number i 
    diamondRecall = zeros(varSize);
    equalityRecall = zeros(varSize);
    diamondTimes = zeros(varSize);
    equalityTimes = zeros(varSize);
end

% Do one sampling for one data set
function [ diamond, euqality, extension] = oneSampling(Variables)
    % 
    [ diamond, equality, extension ] = initVar(Variables);
    MatA = Variables.MatA;
    MatB = Variables.MatB;
    MatC = Variables.MatC;
    valueQuery = Variables.valueQuery;
    knn = Variables.knn;
    % initialize the variables
    [ diamondRecall, diamondTimes, equalityRecall, equalityTimes ] = initVar(varSize);
    % use the average result
    for s = 1:turn
        % temp variables
        [ diamondTemp, equalityTemp, extensionTemp ] = initVar(Variables);
        [ diamondRecalltemp, diamondTimestemp, equalityRecalltemp, equalityTimestemp ] = initVar(varSize);
        for i = 1:length(Variables.samples)
            sample = Variables.samples(i);
            budget = Variables.budget(i);
            [diamondValues, time] = querySampling(MatA', MatB, MatC, budget, sample, knn);
            diamond.time(1,:) = time;
            [equalityValues, time] = queryEqualitySampling(MatA, MatB, MatC, budget, sample, knn);
            equality.time(1,:) = time;
            [extensionValues, time] = queryExtensionSampling(MatA, MatB, MatC, budget, sample, knn);
            extension.time(1,:) = time;
            for n = 1:Variables.NumQueries
                % for each query compute the recall
                diamondTemp.recall(i,n) = sum(diamondValues(1:knn,n) >= valueQuery(knn,n))/knn;
                equalityTemp.recall(i,n) = sum(euqalityValues(1:knn,n) >= valueQuery(knn,n))/knn;
                extensionTemp.recall(i,n) = sum(euqalityValues(1:knn,n) >= valueQuery(knn,n))/knn;
            end
        end
        diamond.recall = diamond.recall + diamondTemp.recall;
        diamond.times = diamond.times + diamondTemp.times;
        
        equality.recall = equality.recall + equalityTemp.recall;
        equality.times = equality.times + equalityTemp.times;
        
        extension.recall = extension.recall + extensionTemp.recall;
        extension.times = extension.times + extensionTemp.times;
    end
    diamond.recall = diamond.recall/Variables.turn;
    diamond.times = diamond.times/Variables.turn;
    
    equality.recall = equality.recall/Variables.turn;
    equality.times = equality.times/Variables.turn;
    
    extension.recall = extension.recall/Variables.turn;
    extension.times = extension.times/Variables.turn;
end

%% draw the time figure
function drawTimeFig(titlename, Variables, diamond, equality, extension)
    fullTime = Variables.fullTime;
    out_dir = Variables.out_dir;
    h = figure; hold on;title(titlename); 
    xlabel('Queries');
    ylabel('log_{10}T(sec)');
    % average time for each query in exhaustive search
    plot(1:NumQueries,log10(fullTime*ones(NumQueries,1)),'-k');
    c = ['g','k','b','r','c'];
    tesc = cell(4,1);
    tesc{1} = 'exhaustive';
    % queries-recall under each samples
    c = ['g','k','b','r','c'];
    for i = 1: length(Variabes.samples)
      sample = Variabes.samples(i);
      h = figure; hold on;title([titlename,'-',num2str(sample)]); 
      xlabel('Queries');
      ylabel('log_{10}T(sec)');
      % average time for each query in exhaustive search
      plot(1:NumQueries,log10(fullTime*ones(NumQueries,1)),'-c');
      tesc = cell(length(samples)*2 + 1,1);
      plot(1:Variables.NumQueries, diamondTimes(i,:),c(1)); 
      plot(1:Variables.NumQueries, equalityTimes(i,:),c(2));
      plot(1:Variables.NumQueries, extensionTimes(i,:),c(3));
      tesc{2} = ['diamond,s=',num2str(sample)]; 
      tesc{3} = ['equality,s=',num2str(sample)]; 
      tesc{4} = ['extension,s=',num2str(sample)]; 
      legend(tesc,4);
      saveas(h,fullfile(out_dir,[titlename,'-',num2str(sample),'.png']));
    end
end

%% recall-queries figure
function drawRecallFig(titlename, out_dir, NumQueries, samples, diamondRecall, equalityRecall)
    h = figure; hold on; title(titlename);
    xlabel('Queries'); 
    ylabel('recall');
    c = ['r','b','k','g', 'c'];
    axis([1 NumQueries 0 1.1]);
    desc = cell(size(samples,2)*2, 1);
    for i= 1:length(samples)
        plot(1:NumQueries,diamondRecall(i,:),c(i),'LineWidth',2);
        plot(log10(samples),equalityRecall(i,:),['--',c(i)],'LineWidth',2); 
        desc{2*i-1} = ['diamond:t=',num2str(samples(i))];
        desc{2*i} = ['equality:t=',num2str(samples(i))];
    end
    legend(desc,4);  
    saveas(h,fullfile(out_dir,[titlename,'.png'])); 

end

function oneDataSet(dataName, out_dir, ...
                    samples, budget, knn, ...
                    A, B, C,...
                    timeQuery,valueQuery,...
                    turn)
    NumQueries = size(A,1);
    varSize = [length(samples),NumQueries];
    [ diamondRecall, diamondTimes, ...
      equalityRecall, equalityTimes ] = oneSampling(varSize, NumQueries, ...
                                                    samples, budget, knn, ...
                                                    A, B, C, ...
                                                    valueQuery, ...
                                                    turn);
    titlename = ['time-samples-',dataName];
    drawTimeFig(titlename, out_dir, NumQueries, samples, timeQuery, diamondTimes, equalityTimes)
    titlename = ['recall-samples-',dataName];
    drawRecallFig(titlename, out_dir, NumQueries, samples, diamondRecall, equalityRecall)
end
