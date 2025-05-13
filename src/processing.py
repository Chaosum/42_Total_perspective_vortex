import os
import random
from matplotlib import pyplot as plt
import mne
import numpy as np
import pywt
from global_variable import *

class Processing:
    def __init__(self):
        self._data = {}
    
    def load_subject_run(self, subject_id: int, run_id: int):
        """
        Charge une seule run (par fichier .edf) pour un sujet donné.
        """
        BASE_PATH = os.path.abspath("./data/MNE-eegbci-data/files/eegmmidb/1.0.0")
        filename = f"{subject_id}R{run_id:02}.edf"
        file_path = os.path.join(BASE_PATH, subject_id, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier EDF introuvable : {file_path}")

        raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto')
        print(f"✅ Run {run_id} chargée pour {subject_id}")
        return raw

    def extract_epochs_for_run(self, raw, run):
        events, event_id = mne.events_from_annotations(raw)
        # On garde uniquement T1 et T2
        filtered_event_id = {k: v for k, v in event_id.items() if k in ["T1", "T2"]}

        if not filtered_event_id:
            return [], []

        epochs = mne.Epochs(
            raw, events,
            event_id=filtered_event_id,
            tmin=-1.0, tmax=2.0,
            baseline=None,
            preload=True
        )

        X = epochs.get_data()
        y = []
        for e in epochs.events:
            code = e[2]
            for k, v in filtered_event_id.items():
                if v == code:
                    y.append(useful_runs[run][k])
        return X, y
    
    def setup_data (self):
        subject = {}
        for i in range(10):
            subject_id = f"S{i + 1:03}"
            subject[subject_id] = {}
            for run, _ in useful_runs.items():
                try:
                    raw_data = self.load_subject_run(subject_id, run)
                    raw_data = raw_data.filter(7., 30., fir_design='firwin')
                    subject[subject_id][run] = self.extract_epochs_for_run(raw_data, run)
                    print(f"✅ {subject_id} run {run} chargée.")
                except Exception as e:
                    print(f"❌ Erreur pour {subject_id} run {run} : {e}")
        return subject

    def extract_features_from_channel(self, signal, wavelet='db4', level=3):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = []
        for c in coeffs:
            features.append(np.mean(np.abs(c)))         # moyenne absolue
            features.append(np.std(c))                  # écart-type
            features.append(np.sum(np.square(c)))       # énergie
        return features

    def extract_features_from_epoch(self, epoch, wavelet='db4', level=3):
        all_features = []
        for channel in epoch:  # channel.shape = (n_times,)
            features = self.extract_features_from_channel(channel, wavelet, level)
            all_features.extend(features)
        return np.array(all_features)


if __name__ == "__main__":
    p = Processing()

    subject = p.setup_data()
    
    X_total = [] # toutes les epochs de tous les runs
    y_total = [] # la liste des labels correspondants
    for subject_id in subject:
        for run_id in subject[subject_id]:
            X, y = subject[subject_id][run_id]
            X_total.extend(X)
            y_total.extend(y)
    X_total = np.array(X_total)  # shape (n_samples, n_channels, n_times)
    X_wavelet = [p.extract_features_from_epoch(epoch) for epoch in X_total]
    X_wavelet = np.array(X_wavelet)

