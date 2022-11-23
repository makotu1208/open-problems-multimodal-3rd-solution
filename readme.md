# open-problems-multimodal-3rd-solution

This repository is a Open Problems - Open-sourced code for the third-order solution of the Multimodal Single-Cell Integration competition. 
<br>
<br>

## basic usage 
  
 Case 1: quick prediction(use pretrained model and prepared feature)
  1. put into data and model  
       model folder: https://www.kaggle.com/datasets/mhyodo/open-problems-models  
       input/features folder: https://www.kaggle.com/datasets/mhyodo/open-problems-features
       
  2. run this `cd ./code/4.model/pred/ && python prediction.py`

 Case 2: retrain(with prepared feature)
  1. put into data and model  
       model folder: https://www.kaggle.com/datasets/mhyodo/open-problems-models  
       input/features folder: https://www.kaggle.com/datasets/mhyodo/open-problems-features  
       input/fold folder: https://www.kaggle.com/datasets/mhyodo/open-problems-cite-folds  
       
  2. run this  
        `cd ./code/4.model/train/cite/ && python cite-mlp.py`  
        `cd ./code/4.model/train/cite/ && python cite-catboost.py`  
        `cd ./code/4.model/train/multi/ && python multi-mlp.py`  
        `cd ./code/4.model/train/multi/ && python multi-catboost.py`  

 Case 3: make feature
   1. put into data and model 
        input/raw folder:
            https://www.kaggle.com/competitions/open-problems-multimodal/data  
            https://www.kaggle.com/datasets/ryanholbrook/open-problems-raw-counts  
   2. run this  
        `cd ./code/1.raw_to_preprocess/cite/ && python make-cite-sparse-matrix.py`  
        `cd ./code/1.raw_to_preprocess/cite/ && python make-clusters.py`  
        `cd ./code/1.raw_to_preprocess/multi/ && python make-multi-sparse-matrix.py`  
        `cd ./code/1.raw_to_preprocess/cite/ && python make-cite-sparse-matrix.py`  
        `cd ./code/2.preprocess_to_feature/cite/ && python make-base-feature.py`  
        `cd ./code/2.preprocess_to_feature/cite/ && python make-base-word2vec.py`  
        `cd ./code/2.preprocess_to_feature/cite/ && python make-features.py`  
        `cd ./code/2.preprocess_to_feature/multi/ && python multi-allsample-basefeature.py`  
        `cd ./code/2.preprocess_to_feature/multi/ && python multi-allsample-target.py`  

<br>
<br>

[notes]  
Since the creation of features, especially clustering-related features, is quite time-consuming(about 1 day),    
it is recommended to use only the prediction part if the results are to be reproduced (Case 1).  
Also, if you have a machine with little memory, it may stop running in the middle of the process. I am running on a 128GB machine.   
And there may be errors in the dependencies of the packages used. (Please check the requirements file.)  
