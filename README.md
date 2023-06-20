# Introduction
MullBind is an accurate predictor for identifying nucleic-acid-binding residues using multiple-task strategy and large-scale language model. Here, the informative sequence representations and protein structures are generated through the pretrained language model and ESMFold first. Then MullBind employs geometric vector perceptron to extract geometric and relational characteristics from predicted protein structures, as well as leverages the multiple-task deep learning strategy to obtain common binding characteristics from different nucleic acids. Finally, two nucleic-acid-specific fully connected layers are employed to learn the binding patterns of particular nucleic acids. Through comprehensive tests on DNA/RNA benchmark datasets, MullBind was shown to surpass the latest sequence-based methods and even the state-of-the-art structure-based methods. 
![image](https://github.com/songyidong-true/nucleic-acid-binding/blob/main/IMG/MullBind_framework.png)
# System requirement
python 3.8.16  
numpy 1.24.2  
pyg-lib 0.1.0+pt113cu116  
pyparsing 3.0.9  
scikit-learn 1.2.2  
six 1.16.0  
torch 1.13.1+cu116  
torch-cluster 1.6.1+pt113cu116  
torch-geometric 2.2.0   
torch-scatter 2.1.1+pt113cu116  
torch-sparse 0.6.17+pt113cu116  
torch-spline-conv 1.2.2+pt113cu116  
torchaudio 0.13.1+cu116  
torchvision 0.14.1+cu116  
urllib3  1.26.15  
wheel 0.38.4  
# Pretrained language model
You need to prepare the pretrained language model ProtTrans to run MullBind:  
Download the pretrained ProtT5-XL-UniRef50 model ([guide](https://github.com/agemagician/ProtTrans)).  
# ESMFold
The protein structures should be predicted by ESMFold to run MullBind:  
Download the ESMFold model ([guide](https://github.com/facebookresearch/esm))  
# Run MullBind for prediction
Simply run:  
```
python predict.py --dataset_path ../Example/structure_data/ --feature_path ../Example/prottrans/ --input_path ../Example/demo.pkl
```
And the prediction results will be saved in  
```
../Example/results
```
# Dataset and model
We provide the datasets and the trained MullBind models here for those interested in reproducing our paper. The datasets used in this study are stored in ```../Dataset/```.
The trained MullBind models can be found under ```../Model/```.
# contact
Yidong Song (songyd6@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)


