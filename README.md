# Celebrity-Face-Recognition
Celebrity Face Recognition Using Classical Machine Learning
th untitled.ipynb file has all the technics that i've applied on the images such as 

Face detection using OpenCV's Haar Cascade

Feature engineering via:

HOG (Histogram of Oriented Gradients) for texture & shape

Color Histograms for color distribution

Data preprocessing with StandardScaler for feature normalization

Dimensionality reduction using PCA to improve performance and reduce complexity

Model training with multiple algorithms:

Support Vector Machine (SVM) — surprisingly the best performer

XGBoost

RandomForest, AdaBoost, KNN — all tuned via GridSearchCV

Ensemble learning with a Voting Classifier combining multiple models for robustness.

i did all this in the untitled file to be able to create the prepared function for the prediction that the prediciton is based on on the app.py file 



