# STAD68 Project
## Dependencies
- python 3
- tensorflow
- numpy
- matplotlib
- scikit-learn

## Preparing data
- download data from https://www.kaggle.com/dalpozz/creditcardfraud and extract 
- run read_data.py to store data in numpy arrays

## Training 
### Neural Net with Cross-entropy 
- run FCNN_softmax.py
### Neural Net with SVM layer
- run FCNN_SVM.py
### Random Forest
- run random_forest.py. Note this script does both training and testing

## Performance on test set
- For the neural networks run NN_perform.py
- random_forest.py prints average precision score and plot precision-recall curve
