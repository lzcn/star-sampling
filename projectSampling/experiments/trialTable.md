## This is a reference table for experiments

#### Experiment TRIAL001
- Content: __Recall and Time of each Sampling Method__  
- Data Set: 
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_
    + _random data_
    
- Aspects to be assessed
    + Diamond Sampling vs Equality Sampling
      * Recall for different number of samples with maximum budget
      * Recall for different number of samples under the 1k budget
      * Recall for different number of samples under the 10k budget
    + Diamond Sampling vs Extension Sampling
      * Recall for different number of samples with maximum budget
      * Recall for different number of samples under the 1k budget
      * Recall for different number of samples under the 10k budget
    + Time consuming of Diamond Sampling, Equality Sampling and Extension Sampling

#### Experiment TRIAL002
- Content: __Sampling for Query__
- Data Set:
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_
- Aspects to be assessed 
    + Diamond Sampling
      + recall for each query
      + time consuming for each query
    + Equality Sampling 
      + recall for each query
      + time consuming for each query
    + Extension Sampling 
      + recall for each query
      + time consuming for each query
    + Comparison time consuming and recall 
      between the diamond equality and extension method
    + Comparison time consuming and recall 
    between the method with pool and the other without
        * Equality
        * Extension
        * Diamond

#### Experiment TRIAL003

- Content: __Demand Number of Samples__
- Data Set:
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_    
- Aspects to be assessed
    +  Demand number of samples for each data set and each top-t
      * draw the figure of _number_ -- _top-t_ for different method
    +  Accuracy and MSE of each method

#### Experiment TRIAL004

- Content: __Evaluate the ability to keep ordering__
- Data Set:
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_
- Aspects to be assessed
    + Equality Sampling under different scores
    + Diamond Sampling
    + Extension Sampling under different scores
    
#### Experiment TRIAL005

- Content: __The probability of each top value__
- Data Set:
    + _MovieLens:ml-10m_
    + _MovieLens:ml-20m_
    + _hetrec2011-delicious-2k_
    + _hetrec2011-lastfm-2k_
    + _hetrec2011-movielens-2k-v2_
    + _random data_
- Aspects to be assessed
    + The difference probability of each top value between top methods
