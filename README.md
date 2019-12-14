# Learning to smell for wellnesss


This repository contains the code to reproduce the results in the paper "Learning to smell for wellnesss", Datasets can be downloaded from the links described in the paper. The code here replicates the result in 1(1−5) − 2 in table 1 of the paper. Same logic goes for the rest of the results.


# Dependencies
Tensorflow==1.5.0. 
Keras==2.1.2. 
Python==3.5.4. 
scikit-learn==0.19.1. 

# Training and Testing
python eval.py will train the model and test the target data over 5 iterations.
Note that the result may be a bit higher or lower than the one reported in the paper due to some randomness issues but should converge to that value over longer iterations.

# Citation
@article{owoeye2019learning,  
  title={Learning to smell for wellness},  
  author={Owoeye, Kehinde},  
  journal={NeurIPS Joint Workshop on AI for Social Good},  
  year={2019}. 
}
