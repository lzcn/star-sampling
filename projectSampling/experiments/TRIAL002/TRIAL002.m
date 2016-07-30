function TRIAL002(data_path, out_dir, samples, budget, knn, turn)
    
     Variables.samples = samples;
     Variables.out_dir = out_dir;
     Variables.budget = budget;
     Variables.turn = turn;
     Variables.knn = knn;
     % set data path
     ml_10m_path = fullfile(data_path,'MovieLens','ml-10m');
     ml_20m_path = fullfile(data_path,'MovieLens','ml-20m');
     ml_2k_path = fullfile(data_path, 'hetrec2011-movielens-2k-v2');
     lastfm_path = fullfile(data_path, 'hetrec2011-lastfm-2k');
     delicious_path = fullfile(data_path, 'hetrec2011-delicious-2k');
     
     %% ml-10m
     Variables.dataName = 'ml-10m';
     
     load(fullfile(ml_10m_path,'User.Mat'));
     load(fullfile(ml_10m_path,'Movie.Mat'));
     load(fullfile(ml_10m_path,'Tag.Mat'));
     load(fullfile(ml_10m_path,'valueQuery.Mat'));
     load(fullfile(ml_10m_path,'timeQuery.Mat'));
     Variables.MatA = User;
     Variables.MatB = Movie;
     Variables.MatC = Tag;
     Variables.valueQuery = valueQuery;
     Variables.timeQuery = timeQuery;
     Variables.NumQueries = size(User,1);
     oneDataSet(Variables);
     
     %% ml-20m
     Variables.dataName = 'ml-20m';
 
     load(fullfile(ml_20m_path,'User.Mat'));
     load(fullfile(ml_20m_path,'Movie.Mat'));
     load(fullfile(ml_20m_path,'Tag.Mat'));
     load(fullfile(ml_20m_path,'valueQuery.Mat'));
     load(fullfile(ml_20m_path,'timeQuery.Mat'));
 
     Variables.MatA = User;
     Variables.MatB = Movie;
     Variables.MatC = Tag;
     Variables.valueQuery = valueQuery;
     Variables.timeQuery = timeQuery;
     Variables.NumQueries = size(User,1);
     oneDataSet(Variables);
     
     %% ml-2k
     Variables.dataName = 'ml-2k';
 
     load(fullfile(ml_2k_path,'User.Mat'));
     load(fullfile(ml_2k_path,'Movie.Mat'));
     load(fullfile(ml_2k_path,'Tag.Mat'));
     load(fullfile(ml_2k_path,'valueQuery.Mat'));
     load(fullfile(ml_2k_path,'timeQuery.Mat'));
 
     Variables.MatA = User;
     Variables.MatB = Movie;
     Variables.MatC = Tag;
     Variables.valueQuery = valueQuery;
     Variables.timeQuery = timeQuery;
     Variables.NumQueries = size(User,1);
     oneDataSet(Variables);
     
     %% lastfm
 
     Variables.dataName = 'lastfm';
 
     load(fullfile(lastfm_path,'User.Mat'));
     load(fullfile(lastfm_path,'Artist.Mat'));
     load(fullfile(lastfm_path,'Tag.Mat'));
     load(fullfile(lastfm_path,'valueQuery.Mat'));
     load(fullfile(lastfm_path,'timeQuery.Mat'));

     Variables.MatA = User;
     Variables.MatB = Artist;
     Variables.MatC = Tag;
     Variables.valueQuery = valueQuery;
     Variables.timeQuery = timeQuery;
     Variables.NumQueries = size(User,1);
     oneDataSet(Variables);
     
      
     %% delicious
     Variables.dataName = 'delicious';
 
     load(fullfile(delicious_path,'User.Mat'));
     load(fullfile(delicious_path,'Url.Mat'));
     load(fullfile(delicious_path,'Tag.Mat'));
     load(fullfile(delicious_path,'valueQuery.Mat'));
     load(fullfile(delicious_path,'timeQuery.Mat'));
     
     Variables.MatA = User;
     Variables.MatB = Url;
     Variables.MatC = Tag;
     Variables.valueQuery = valueQuery;
     Variables.timeQuery = timeQuery;
     Variables.NumQueries = size(User,1);
     oneDataSet(Variables);
     
end

%% Initialize Variables
function [ diamond, equality, extension ] = initVar(Variables)
    varSize = [Variables.NumQueries, length(Variables.samples)];
    diamond.recall = zeros(varSize);
    diamond.times = zeros(varSize);
    equality.recall = zeros(varSize);
    equality.times = zeros(varSize);
    extension.recall = zeros(varSize);
    extension.times = zeros(varSize);
end

% Do one sampling for one data set
function [ diamond, equality, extension] = oneSampling(Variables)
    % 
    [ diamond, equality, extension ] = initVar(Variables);
    turn = Variables.turn;
    MatA = Variables.MatA;
    MatB = Variables.MatB;
    MatC = Variables.MatC;
    valueQuery = Variables.valueQuery;
    knn = Variables.knn;
    % use the average result
    for s = 1:turn
        % temp variables
        [ diamondTemp, equalityTemp, extensionTemp ] = initVar(Variables);
        for i = 1:length(Variables.samples)
            sample = Variables.samples(i);
            budget = Variables.budget(i);
            [diamondValues, time] = querySampling(MatA', MatB, MatC, budget, sample, knn);
            diamond.time(:, i) = time;
            [equalityValues, time] = queryEqualitySampling(MatA, MatB, MatC, budget, sample, knn);
            equality.time(:, i) = time;
            [extensionValues, time] = queryExtensionSampling(MatA, MatB, MatC, budget, sample, knn);
            extension.time(:,i) = time;
            for n = 1:Variables.NumQueries
                % for each query compute the recall
                diamondTemp.recall(n,i) = sum(diamondValues(1:knn,n) >= valueQuery(knn,n))/knn;
                equalityTemp.recall(n,i) = sum(equalityValues(1:knn,n) >= valueQuery(knn,n))/knn;
                extensionTemp.recall(n,i) = sum(extensionValues(1:knn,n) >= valueQuery(knn,n))/knn;
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
    NumQueries = Variables.NumQueries;
    fullTime = Variables.fullTime;
    out_dir = Variables.out_dir;
    % queries-recall under each samples
    for i = 1: length(Variabes.samples)
      sample = Variabes.samples(i);
      h = figure; hold on;
      title([titlename,'-',num2str(sample)]); 
      xlabel('queries');
      ylabel('log_{10} t(s)');
      % average time for each query in exhaustive search
      plot(1:NumQueries,log10(fullTime*ones(NumQueries,1)),'c');
      plot(1:Variables.NumQueries, diamond.times(:,i),'--b'); 
      plot(1:Variables.NumQueries, equality.times(:,i),'--r');
      plot(1:Variables.NumQueries, extension.times(:,i),'--g');
      legend('exhaustive','diamond','equality','extension',4);
      saveas(h,fullfile(out_dir,[titlename,'-',num2str(sample),'.pdf']));
    end
end

%% recall-queries figure
function drawRecallFig(titlename, Variables, diamond, equality, extension)
    out_dir = Variables.out_dir;
    NumQueries = Variables.NumQueries;
    for i= 1:length(samples)
        h = figure; hold on;
        sample = Variables.samples(i);
        title([titlename,'-',num2str(sample)]);
        xlabel('queries'); 
        ylabel('recall');
        axis([1 NumQueries 0 1.1]);
        plot(1:NumQueries, diamond.recall(:,i),'--b');
        plot(1:NumQueries, equality.recall(:,i),'--r'); 
        plot(1:NumQueries, extension.recall(:,i),'--g'); 
        legend('diamond','equality','extension',4);
        saveas(h,fullfile(out_dir,[titlename,'-',num2str(sample),'.pdf'])); 
    end
end

function oneDataSet(Variables)
    [ diamond, equality, extension] = oneSampling(Variables);
    titlename = ['time-samples-',Variables.dataName];
    drawTimeFig(titlename, Variables, diamond, equality, extension);
    titlename = ['recall-samples-',Variables.dataName];
    drawRecallFig(titlename, Variables, diamond, equality, extension);
end
