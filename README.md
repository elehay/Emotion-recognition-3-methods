# Emotion-recognition-3-methods
A emotion recognition app made using Python 3.11.9. and copilot (Claude sonnet 3.5)

The HOG + SVM Part doesn't show the confidence for performance reasons. If you wish to see them, change the SVN model "LinearSVN" to "SVN" and add the argument "probability=true". This will significatively slow down fitting the hog into the model. You might also need to change the display function accordingly.

The kNN part is taken from part one of this projet : https://github.com/elehay/Emotion-recognition-knn-lbp

The pre-trainned cnn model was found here : https://github.com/nmfadil/FER-Pretrained-MiniXception.git
