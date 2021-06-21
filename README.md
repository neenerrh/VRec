#Video Recommendation using Knowledge-Graph-Embedding
This is the code of a knowledge graph embedding framework – RKGE – with a novel recurrent network architecture for high-quality recommendation. RKGE [1] not only learns the semantic representation of different types of entities but also automatically captures entity relations encoded in KGs.


## Pre-requisits

- ### Running environment

  - Python 3
  
  - Pytorch (conda 4.5.11 - https://zhuanlan.zhihu.com/p/26854386)
  

- ### Datasets - XuatengX 


   

## Modules

    

  - ### Data Split (data-split.py)
  
    - Split the user-movie rating data into training and test data

      - Input Data: rating-delete-missing-itemid.txt

      - Output Data: training.txt, test.txt


  - ### Negative Sample (negative-sample.py)
  
    - Sample negative movies for each user to balance the model training 
  
      - Input Data: training.txt

      - Output Data: negative.txt


  - ### Path Extraction （path-extraction.py）
  
    - Extract paths for both positive and negative user-moive interaction
    
      - Input Data: training.txt, negative.txt, auxiliary-mapping.txt,
      
      - Output Data: positive-path.txt, negative-path.txt
      
  
  - ### Recurrent Neural Network (recurrent-neural-network.py)
  
    - Feed both postive and negative path into the recurrent neural network, train and evaluate the model
    
      - Input Data: positive-path.txt, negative-path.txt, training.txt, test.txt, pre-train-user-embedding.txt, pre-train-movie-embedding.txt (To speed up model training process, the user and movie embedding is pre-trained via [2]. You may also use matrix factorization [3] or bayesain personalized ranking [4] to pre-train the embeddings). 
      
      - Output Data: results.txt
      

