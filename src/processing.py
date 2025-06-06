import shutil
import os
from matplotlib import pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci
from global_variable import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib
from waveletsTransformer import WaveletTransformer

class Processing:
    def __init__(self):
        self._data = {}
    
    def load_subject_run(self, subject_id: str, run_id: int):
        """
        Charge une seule run (par fichier .edf) pour un sujet donné.
        """
        BASE_PATH = os.path.abspath("./src/data/MNE-eegbci-data/files/eegmmidb/1.0.0")
        filename = f"{subject_id}R{run_id:02}.edf"
        file_path = os.path.join(BASE_PATH, subject_id, filename)

        if not os.path.exists(file_path):
            subj_num = int(subject_id[1:])  # 'S014' -> 14
            print(f"📥 Téléchargement du sujet {subject_id}, run {run_id}...")
            mne.set_config("MNE_DATA", os.path.abspath("./data/"))
            eegbci.load_data(subj_num, [run_id])

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
        if len(set(y)) < 2:
            print(f"⚠️ Run {run} ignoré : seulement une classe présente.")
            return [], []
        return X, y
    
    def setup_data (self, subject_id=None, run_id=None):
        subject = {}
        if subject_id is None and run_id is None:
            for i in range(109):
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
        else:
            subject_id = f"S{subject_id :03}"
            subject[subject_id] = {}
            try:
                raw_data = self.load_subject_run(subject_id, run_id)
                raw_data = raw_data.filter(7., 30., fir_design='firwin')
                subject[subject_id][run_id] = self.extract_epochs_for_run(raw_data, run_id)
                print(f"✅ {subject_id} run {run_id} chargée.")
            except Exception as e:
                print(f"❌ Erreur pour {subject_id} run {run_id} : {e}")
        return subject
    
    def get_all_data(self):
        subject = self.setup_data()
        
        X_total = []
        y_total = []
        for subject_id in subject:
            for run_id in subject[subject_id]:
                X, y = subject[subject_id][run_id]
                X_total.extend(X)
                y_total.extend(y)

        X_total = np.array(X_total)
        y_total = np.array(y_total)
        return X_total, y_total
    
    def get_all_data(self, subject_id, run_id):
        subject = self.setup_data(subject_id, run_id)
        
        X_total = []
        y_total = []
        for subject_id in subject:
            for run_id in subject[subject_id]:
                X, y = subject[subject_id][run_id]
                X_total.extend(X)
                y_total.extend(y)

        X_total = np.array(X_total)
        y_total = np.array(y_total)
        return X_total, y_total