function TRIAL003(data_path, out_dir)

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
    load(fullfile(ml_10m_path,'topValue.Mat'));
    load(fullfile(ml_10m_path,'fullTime.Mat'));
    load(fullfile(ml_10m_path,'topIndexes.Mat'));
    [score1,actual1,weight1] = diamondExp003(User',Movie,Tag,1e8);
    [score2,actual2,weight2] = equalityExp003(User',Movie,Tag,1e8);

end
