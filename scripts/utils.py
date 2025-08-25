import matplotlib.pyplot as plt
import seaborn as sns 
import json
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)

def plot_confusion_matrix(y_true, y_pred, normalize=False, labels=None):
    cm = confusion_matrix(
        y_true, y_pred,
        labels=labels,
        normalize='true' if normalize else None
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()


def plot_roc_per_class(y_true, y_proba, class_names):
   
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    plt.figure()
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC per class (OvR)"); plt.legend(loc="lower right")
    plt.grid(True); plt.tight_layout(); plt.show()

def print_metrics_table(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    f1   = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

def save_metrics_to_json(y_true, y_pred, filepath="results/multiclass_metrics.json"):
    import os, json
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall":    recall_score(y_true, y_pred, average="macro"),
        "f1":        f1_score(y_true, y_pred, average="macro"),
    }
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")

    
