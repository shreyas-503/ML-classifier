# Infant Cry Classification using Machine Learning and Deep Learning

This project implements an end-to-end machine learning and deep learning system to classify infant cries into five categories: hungry, tired, discomfort, belly pain, and burping. The pipeline includes audio preprocessing, feature extraction, model training, evaluation, and deployment using a Gradio web interface.

## Key Features

- Audio feature extraction using MFCC, Chroma, Mel-Spectrogram, and Spectral Contrast  
- Machine learning models: KNN, SVM, Decision Tree, Random Forest, XGBoost  
- Deep learning model: CNN trained on Mel-Spectrogram images  
- Model evaluation using Accuracy, Macro F1, Balanced Accuracy, and Confusion Matrices  
- Interactive Gradio interface for real-time prediction and model performance visualization

# Dataset Structure

Total Samples: 680

1.  hungry        364
2.  discomfort     99
3.  belly_pain     91
4.  tired          69
5.  burping        57


# Model Performance Summary

1. KNN 0.7206
2. Decision Tree 0.8382
3. Random Forest	0.98
4. XGBoost	0.94
5. SVM	0.91
6. CNN	0.93


# Tech Stack

  Python, Librosa, Scikit-learn, XGBoost, TensorFlow, Keras, Gradio, NumPy, Pandas, Matplotlib, Seaborn

## How to Run

1. Mount Google Drive in Colab  
2. Install required dependencies  
3. Set the DATA_PATH variable  
4. Run the main script to train all models and launch the Gradio interface  
