function TRIAL005(data_path,out_dir)
    % set data path
    ml_10m_path = fullfile(data_path,'MovieLens','ml-10m');
    ml_20m_path = fullfile(data_path,'MovieLens','ml-20m');
    ml_2k_path = fullfile(data_path, 'hetrec2011-movielens-2k-v2');
    lastfm_path = fullfile(data_path, 'hetrec2011-lastfm-2k');
    delicious_path = fullfile(data_path, 'hetrec2011-delicious-2k');
    randData_path = fullfile(data_path, 'random');
    
    % set variables
    Variables.topIndexes = [];
    Variables.out_dir = out_dir;
    %% ml-10m
    Variables.dataName = 'ml-10m';

    load(fullfile(ml_10m_path,'User.Mat'));
    load(fullfile(ml_10m_path,'Movie.Mat'));
    load(fullfile(ml_10m_path,'Tag.Mat'));
    load(fullfile(ml_10m_path,'topValue.Mat'));
    load(fullfile(ml_10m_path,'fullTime.Mat'));
    load(fullfile(ml_10m_path,'topIndexes.Mat'));
    
    Variables.MatA = User;
    Variables.MatB = Movie;
    Variables.MatC = Tag;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);
    %% ml-20m
    Variables.dataName = 'ml-20m';
 
    load(fullfile(ml_20m_path,'User.Mat'));
    load(fullfile(ml_20m_path,'Movie.Mat'));
    load(fullfile(ml_20m_path,'Tag.Mat'));
    load(fullfile(ml_20m_path,'topValue.Mat'));
    load(fullfile(ml_20m_path,'fullTime.Mat'));
    load(fullfile(ml_20m_path,'topIndexes.Mat'));
 
    Variables.MatA = User;
    Variables.MatB = Movie;
    Variables.MatC = Tag;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);

    %% ml-2k
    Variables.dataName = 'ml-2k';

    load(fullfile(ml_2k_path,'User.Mat'));
    load(fullfile(ml_2k_path,'Movie.Mat'));
    load(fullfile(ml_2k_path,'Tag.Mat'));
    load(fullfile(ml_2k_path,'topValue.Mat'));
    load(fullfile(ml_2k_path,'fullTime.Mat'));
    load(fullfile(ml_2k_path,'topIndexes.Mat'));

    Variables.MatA = User;
    Variables.MatB = Movie;
    Variables.MatC = Tag;
    Variables.topIndexes = topIndexes;   

    oneDataSet(Variables);
    %% lastfm 
    Variables.dataName = 'lastfm';

    load(fullfile(lastfm_path,'User.Mat'));
    load(fullfile(lastfm_path,'Artist.Mat'));
    load(fullfile(lastfm_path,'Tag.Mat'));
    load(fullfile(lastfm_path,'topValue.Mat'));
    load(fullfile(lastfm_path,'fullTime.Mat'));
    load(fullfile(lastfm_path,'topIndexes.Mat'));
 
    Variables.MatA = User;
    Variables.MatB = Artist;
    Variables.MatC = Tag;
    Variables.topIndexes = topIndexes;
     
    oneDataSet(Variables);
     
    %% delicious
    Variables.dataName = 'delicious';
 
    load(fullfile(delicious_path,'User.Mat'));
    load(fullfile(delicious_path,'Url.Mat'));
    load(fullfile(delicious_path,'Tag.Mat'));
    load(fullfile(delicious_path,'topValue.Mat'));
    load(fullfile(delicious_path,'fullTime.Mat'));
    load(fullfile(delicious_path,'topIndexes.Mat'));
 
    Variables.MatA = User;
    Variables.MatB = Url;
    Variables.MatC = Tag;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);

    %% random data
    Variables.dataName = 'random';
    
    load(fullfile(randData_path, 'MatA.Mat'));
    load(fullfile(randData_path, 'MatB.Mat'));
    load(fullfile(randData_path, 'MAtC.Mat'));
    load(fullfile(randData_path, 'topValue.Mat'));
    load(fullfile(randData_path, 'fullTime.Mat'));
    load(fullfile(randData_path, 'topIndexes.Mat'));
    Variables.MatA = MatA;
    Variables.MatB = MatB;
    Variables.MatC = MatC;
    Variables.topIndexes = topIndexes;
    
    oneDataSet(Variables);  
end

function oneDataSet(Variables)
    [p1,p2,p3] = getProbability(Variables.MatA,Variables.MatB,Variables.MatC,double(Variables.topIndexes));
    h = figure; hold on;
    filename = ['probability','-',Variables.dataName];
    title(filename);
    plot(p1,'c');
    plot(p2,'b');
    plot(p3,'r');
    legend('diamond','equality','extension');
    saveas(h,fullfile(Variables.out_dir,[filename,'.pdf']));
end
