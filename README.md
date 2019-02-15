# plant_classification

This is code for classification of plant images into following 12 categoris

Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
------------------
#Model building
plant_classification_model_building.py
Above model building uses InceptionV3 network.


#Model evaluation
plant_classification_evaluation.py
Accuracy achieved is 96%
Confusion matrix

[[20  0  0  0  1  0  7  0  0  0  0  0]
 [ 0 44  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 27  0  0  0  0  0  0  0  0  0]
 [ 0  0  1 59  0  0  0  0  1  0  0  0]
 [ 0  0  0  0 18  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 50  0  0  0  0  0  2]
 [ 5  0  0  0  0  0 54  0  0  0  0  0]
 [ 0  0  0  0  0  0  0 25  0  0  0  0]
 [ 0  0  0  1  0  0  0  0 52  4  0  1]
 [ 0  0  1  0  0  0  0  0  2 17  0  0]
 [ 0  0  0  0  0  0  0  0  0  0 48  0]
 [ 0  0  0  0  0  0  0  0  0  0  0 35]]

