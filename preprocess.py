import os
import numpy as np

def list_txt_files(folder, limit=None):
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    ])
    return files[:limit] if limit else files

def load_eeg_file(file_path, segment_length=None):
    try:
        data = np.genfromtxt(file_path)
        if segment_length:
            data = data[:segment_length]
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def load_eeg_folder(folder, limit=None, segment_length=None):
    eeg_data = []
    files = list_txt_files(folder, limit)
    for filename in files:
        file_path = os.path.join(folder, filename)
        data = load_eeg_file(file_path, segment_length)
        if data is not None:
            eeg_data.append((filename, data))
    return eeg_data
