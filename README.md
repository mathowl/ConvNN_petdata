# ConvNN_petdata
Basic neural net optimization using Keras for the Kaggle cat-dog data set.  (My first Keras NN ^-^ )

## Data
The Kaggle cats and dogs data set can be downloaded from here https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

## Data processing
CONV-preproc_Petdata.py builds the features and labels. Preprocessing fails on certain images but the resulting set is sufficiently large for training

## Training neural net
The architecture is Sequential of the form Features->ConV->MaxPooling->Dense->Labels. The CONV-ArchOpt varies over dense layers, conv layers and layer sizes. Optimization results can be viewed using tensorboard.

## Classification using the trained net
CONV-predict is used to classify a given image using the optimal neural net

