import numpy as np
from preprocess import load_eeg_folder
from extract_features import extract_all_features
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Z=0, F=1, S=2
def build_dataset_three_folders(z_folder, f_folder, s_folder, limit=100, fs=256):
    X, y = [], []
    feature_names = None
    for _, sig in load_eeg_folder(z_folder, limit=limit):
        feats, names = extract_all_features(sig, fs=fs)
        feature_names = feature_names or names
        X.append(feats); y.append(0)
    for _, sig in load_eeg_folder(f_folder, limit=limit):
        feats, _ = extract_all_features(sig, fs=fs)
        X.append(feats); y.append(1)
    for _, sig in load_eeg_folder(s_folder, limit=limit):
        feats, _ = extract_all_features(sig, fs=fs)
        X.append(feats); y.append(2)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), feature_names

def train_and_eval(
    z_folder="bonn-eeg-ml/z (2)/z/Z",
    f_folder="bonn-eeg-ml/f/F",
    s_folder="bonn-eeg-ml/s/S",
    limit=100, fs=256, seed=42
):
    X, y, feature_names = build_dataset_three_folders(z_folder, f_folder, s_folder, limit=limit, fs=fs)
    class_names = ["Z", "F", "S"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        clf_cv = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        clf_cv.fit(X_train[tr], y_train[tr])
        print(f"Fold {i} done.")

    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)  

    return clf, (X_test, y_test), y_pred, y_probs, class_names, feature_names

def main():
  
    from sklearn.metrics import accuracy_score
    clf, (X_test, y_test), y_pred, _, _, _ = train_and_eval()
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
