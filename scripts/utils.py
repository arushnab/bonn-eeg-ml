import matplotlib.pyplot as plt
import seaborn as sns 
import json
import os
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)

def plot_confusion_matrix(y_test, y_pred, normalize=False):
    cm = confusion_matrix(y_test, y_pred, normalize = 'true' if normalize else None)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap = 'Blues')
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.show()

def plot_roc_curve(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], '--', color='gray') 
    plt.xlabel("False Posiitve Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()    
    plt.grid(True)
    plt.show()

def print_metrics_table(y_test, y_pred):
    acc= accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}")

def save_metrics_to_json(y_test, y_pred, filepath="results/w4_logreg_zs_metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {filepath}")

    
