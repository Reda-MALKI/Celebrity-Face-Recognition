# Celebrity-Face-Recognition
Celebrity Face Recognition Using Classical Machine Learning
🤖 Celebrity Face Recognition Using Classical Machine Learning
📸 Tech Stack: Python · OpenCV · scikit-learn · SVM · HOG · Haar Cascades · PCA · StandardScaler · Voting Classifier · Flask · HTML/CSS/JS

I'm excited to share a complete ML-powered web app I built that can recognize the face of a celebrity from a photo — achieving 78.68% accuracy using only classical machine learning methods (no deep learning!).

🚀 What I Built:
A web-based image classification system for 5 well-known sports celebrities

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

Ensemble learning with a Voting Classifier combining multiple models for robustness

🖥 User Interface:

Stylish and responsive frontend UI

Live image preview after upload

Instant prediction display on the same page

Smooth experience with no page reloads, built using HTML/CSS and vanilla JavaScript

📈 Results:

Achieved 78.68% accuracy on unseen test images

Deployed a working end-to-end pipeline: from image selection to prediction display

💡 What I Learned:

How to combine HOG + color histograms to simulate deep features

Importance of feature scaling and dimensionality reduction (StandardScaler & PCA) in classical ML workflows

Benefits of ensemble methods like Voting Classifier to boost prediction stability

Building real-time ML apps with Flask

Styling and enhancing UX with animations and interactivity
