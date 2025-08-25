# Bonn EEG ML Project  

Built an end-to-end machine learning pipeline for seizure detection using the Bonn EEG dataset. The work combines advanced signal processing with interpretable machine learning to create a reproducible framework for biosignal classification.  

## Technical Highlights  
- **Signal Processing**: Extracted frequency-domain (EEG bandpowers via Welch PSD) and time-domain features including entropy, Hjorth parameters, line length, and zero crossings to capture complex neural dynamics.  
- **Machine Learning**: Built and optimized classifiers (Logistic Regression, SVM, Random Forest) for both binary and multi-class EEG state prediction (healthy, interictal, seizure).  
- **Evaluation**: Automated computation of accuracy, precision, recall, F1, and ROC/AUC. Produced confusion matrices, ROC curves, and analyses of misclassified EEG signals for model diagnostics.  
- **Interpretability**: Applied Random Forest feature importances and SHAP analysis to highlight neurophysiologically meaningful predictors of seizure activity.  
- **Research Reproducibility**: Modular pipeline that saves trained models, logs results to JSON, and generates standardized diagnostic plots, enabling transparent and repeatable experiments.  

## Dataset  
Single-channel EEG recordings (4097 samples at 173.61 Hz) from the University of Bonn, focusing on Sets Z (healthy), F (interictal), and S (seizure).

> **Note**  
> This repository is still being cleaned up and organized. Some files and documentation may change as the project is refined.

