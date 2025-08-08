import joblib
import os
import numpy as np
from preprocess import load_eeg_folder
from extract_features import extract_bandpowers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def prepare_dataset(z_folder, s_folder, limit=100):
    z_signals = load_eeg_folder(z_folder, limit)
    s_signals = load_eeg_folder(s_folder, limit)
    X, y = [], []
    
    for _, signal in z_signals:
        X.append(list(extract_bandpowers(signal).values()))
        y.append(0)

    for _, signal in s_signals:
        X.append(list(extract_bandpowers(signal).values()))
        y.append(1)
    
    return np.array(X), np.array(y)

def main():
    z_folder = "bonn-eeg-ml/z (2)/z/Z"
    s_folder = "bonn-eeg-ml/s/S"
    X, y = prepare_dataset(z_folder, s_folder)

    print("Total samples:", len(X))
#is np.unqiue really needed here??
    print("Label distribtuion", np.unique(y, return_counts= True))

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify = y)
    
    ##print a whole bunch of yip yap

    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    print("\n--- Cross-validation on Training Data ---")
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        print(f"\nFold {fold} Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred))

    print(f"\nAverage Cross Validation Accuracy: {np.mean(fold_accuracies):.4f}")

    # Final model training on the entire training+validation set 
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_trainval, y_trainval)
    y_final_pred = final_model.predict(X_test)

    print("\n--- Final Evaluation on Test Set ---")
    print("Test Accuracy:", accuracy_score(y_test, y_final_pred))
    print(classification_report(y_test, y_final_pred))

    # Save the final model
    os.makedirs("models", exist_ok = True)
    joblib.dump(final_model, "models/w4_logreg_zs.pkl")

    #Save the test data
    os.makedirs("data", exist_ok=True)
    joblib.dump(X_test, "data/w4_logreg_zs_Xtest.pkl")
    joblib.dump(y_test, "data/w4_logreg_zs_ytest.pkl")

if __name__ == "__main__":
    main()
