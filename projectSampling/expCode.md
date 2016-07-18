## This is a reference table for experiments

#### Experiment #001
- Content: __Diamond Sampling__ vs __Equality Sampling__
- DataSet: 
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_
- Aspects to be assessed
    + Recall for different number of samples with maximum budget
    + Recall for different number of samples under the same budget
    + Time consuming
    + The difference probability of each top value between top methods

#### Experiments #002
- Content: __Sampling for Query__
- DataSet:
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_

- Aspects to be assessed 
    + average recall for each query
    + average time consuming for each query
    + comparison between the method with pool and the other without
        * time consuming
        * recall

#### Experiments #003
- Content: __Demand Number of Samples__
- DataSet:
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_

- Aspects to be assessed
    +  Demand number of samples for each data set and each top-t

#### Experiments #004
- Content: __Evaluate the ability to keep ordering__
- DataSet:
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_
- Aspects to be assessed
    + Equality Sampling
    + Diamond Sampling
