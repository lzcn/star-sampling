function TRIAL002(paths, dataName, out_dir, samples, budget, knn, turn)
     % set variables
     Variables.knn = knn;
     Variables.dataName = '';
     Variables.samples = samples;
     Variables.budget = budget;
     Variables.turn = turn;
     Variables.out_dir = out_dir;
     for n = 1:length(paths)
         path = paths{n};
         % load data
         Variables.dataName = dataName{n};
         load(fullfile(path,'User.Mat'));
         load(fullfile(path,'Item.Mat'));
         load(fullfile(path,'Tag.Mat'));
         load(fullfile(path,'valueQuery.Mat'));
         load(fullfile(path,'timeQuery.Mat'));
         Variables.A = User;
         Variables.B = Item;
         Variables.C = Tag;
         Variables.NumQueries = size(User,1);
         Variables.valueQuery = valueQuery;
         Variables.timeQuery = timeQuery;
         oneDataSet(Variables);
     end
end

%% Initialize Variables
function [ diamond, central, extension ] = initVar(Variables)
    varSize = [Variables.NumQueries, length(Variables.samples)];
    diamond.recall = zeros(varSize);
    diamond.times = zeros(varSize);
    central.recall = zeros(varSize);
    central.times = zeros(varSize);
    extension.recall = zeros(varSize);
    extension.times = zeros(varSize);
end

% Do one sampling for one data set
function [ diamond, central, extension] = oneSampling(Variables)
    % 
    [ diamond, central, extension ] = initVar(Variables);
    turn = Variables.turn;
    MatA = Variables.MatA;
    MatB = Variables.MatB;
    MatC = Variables.MatC;
    valueQuery = Variables.valueQuery;
    knn = Variables.knn;
    % use the average result
    for s = 1:turn
        % temp variables
        [ diamondTemp, centralTemp, extensionTemp ] = initVar(Variables);
        for i = 1:length(Variables.samples)
            sample = Variables.samples(i);
            budget = Variables.budget(i);
            [dValues, time] = queryDiamondSampling(MatA', MatB, MatC, budget, sample, knn);
            diamond.time(:, i) = time;
            [cValues, time] = queryeCentralSampling(MatA, MatB, MatC, budget, sample, knn);
            central.time(:, i) = time;
            [eValues, time] = queryExtensionSampling(MatA, MatB, MatC, budget, sample, knn);
            extension.time(:,i) = time;
            for n = 1:Variables.NumQueries
                % for each query compute the recall
                diamondTemp.recall(n,i) = sum(dValues(1:knn,n) >= valueQuery(knn,n))/knn;
                centralTemp.recall(n,i) = sum(cValues(1:knn,n) >= valueQuery(knn,n))/knn;
                extensionTemp.recall(n,i) = sum(eValues(1:knn,n) >= valueQuery(knn,n))/knn;
            end
        end
        diamond.recall = diamond.recall + diamondTemp.recall;
        diamond.times = diamond.times + diamondTemp.times;
        
        central.recall = central.recall + centralTemp.recall;
        central.times = central.times + centralTemp.times;
        
        extension.recall = extension.recall + extensionTemp.recall;
        extension.times = extension.times + extensionTemp.times;
    end
    diamond.recall = diamond.recall/Variables.turn;
    diamond.times = diamond.times/Variables.turn;
    diamond_recall = diamond.recall;
    diamond_times = diamond.times;
    save(fullfile(out_dir,'diamond_recall.mat'),'diamond_recall');
    save(fullfile(out_dir,'diamond_times.mat'),'diamond_times');
    
    central.recall = central.recall/Variables.turn;
    central.times = central.times/Variables.turn;
    central_recall = central.recall;
    central_times = central.times;
    save(fullfile(out_dir,'central_recall.mat'),'central_recall');
    save(fullfile(out_dir,'central_times.mat'),'central_times');
    
    extension.recall = extension.recall/Variables.turn;
    extension.times = extension.times/Variables.turn;
    extension_recall = extension.recall;
    extension_times = extension.times;
    save(fullfile(out_dir,'extension_recall.mat'),'extension_recall');
    save(fullfile(out_dir,'extension_times.mat'),'extension_times');
end

%% draw the time figure
function drawTimeFig(titlename, Variables, diamond, central, extension)
    NumQueries = Variables.NumQueries;
    timeQuery = Variables.timeQuery;
    out_dir = Variables.out_dir;
    % queries-recall under each samples
    for i = 1: length(Variables.samples)
      sample = Variables.samples(i);
      h = figure; hold on;
      title([titlename,'-',num2str(sample)]); 
      xlabel('queries');
      ylabel('log_{10} t(s)');
      % average time for each query in exhaustive search
      plot(1:NumQueries,log10(timeQuery*ones(NumQueries,1)),'c');
      plot(1:Variables.NumQueries, diamond.times(:,i),'--b'); 
      plot(1:Variables.NumQueries, central.times(:,i),'--r');
      plot(1:Variables.NumQueries, extension.times(:,i),'--g');
      legend('exhaustive','diamond','central','extension',4);
      saveas(h,fullfile(out_dir,[titlename,'-',num2str(sample),'.pdf']));
    end
end

%% recall-queries figure
function drawRecallFig(titlename, Variables, diamond, central, extension)
    out_dir = Variables.out_dir;
    NumQueries = Variables.NumQueries;
    for i= 1:length(Variables.samples)
        h = figure; hold on;
        sample = Variables.samples(i);
        title([titlename,'-',num2str(sample)]);
        xlabel('queries'); 
        ylabel('recall');
        axis([1 NumQueries 0 1.1]);
        plot(1:NumQueries, diamond.recall(:,i),'--b');
        plot(1:NumQueries, central.recall(:,i),'--r'); 
        plot(1:NumQueries, extension.recall(:,i),'--g'); 
        legend('diamond','central','extension',4);
        saveas(h,fullfile(out_dir,[titlename,'-',num2str(sample),'.pdf'])); 
    end
end

function oneDataSet(Variables)
    [ diamond, central, extension] = oneSampling(Variables);
    titlename = ['time-samples-',Variables.dataName];
    drawTimeFig(titlename, Variables, diamond, central, extension);
    titlename = ['recall-samples-',Variables.dataName];
    drawRecallFig(titlename, Variables, diamond, central, extension);
end
