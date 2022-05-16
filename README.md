### Feature Engineering
Training and testing datasets [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) from Kaggle are used in this project. For feature preprocessing, custom class Preprocessing is created, which includes the following options for scaling, encoding and modification:
 - Encoding of categorical features: frequency, ordinal or no encoding
 - Scaling of numerical features: standard, min_max or no scaling
 - Outliers removal: true or false
 - Conversion of negative numerical values into positive: true or false

### Model Training
Logistic Regression, SVM Linear Classifier, Extra Trees Classifier, Random Forest Classifier, Voting Classifier, Adaptive Boosting Classifier and Deep Neural Network are used for training. The following experiments are created in MLflow:
<img src="./images/experiments.png" width="180">

 - Logistic Regression, SVM Linear Classifier, Extra Trees Classifier, Random Forest Classifier:<br>
Automatic hyperparameters tuning by means of RandomizedSearchCV is implemented to determine best hyperparameters for the four machine learning models. Nested cross validation is employed to select and evaluate the models. F1 score is used as optimization metric for hyperparameters selection. Generalized performance of the classifier is evaluated with three metrics - precision, recall and f1 score.
<img src="./images/tuning.png" width="900">

 - Voting Classifier:<br>
Voting classifier is based on four previously tuned models - Logistic Regression, SVM Linear Classifier, Extra Trees Classifier, Random Forest Classifier. 
<img src="./images/voting.png" width="900">

 - Adaptive Boosting Classifier:<br>
Adaptive Boosting Classifier is implemented with Decision Tree model under the hood. RandomizedSearchCV is used for hyperparameters tuning. F1 score is used as optimization metric for hyperparameters selection.
<img src="./images/boosting.png" width="900">

 - Neural Network:<br>
Deep neural networks with different number of layers and inner layers nodes are trained. For inner layers, relu activation function is used, for outer layer - softmax. For optimization, sparse categorical cross entropy and accuracy metric are chosen.
<img src="./images/neural_network.png" width="900">

### Score on Kaggle
Best score on testing dataset is obtained with Extra Trees Classifier model.
<img src="./images/kaggle_score.png" width="900">


