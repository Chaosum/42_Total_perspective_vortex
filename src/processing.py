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
from scipy.signal import resample

# Réduire la verbosité de MNE
mne.set_log_level('WARNING')

class Processing:
    def __init__(self):
        self._data = {}
    
    """    
    Load la donnée pour un sujet et une run donnés.
    Parameters
    ----------
    subject_id : str
        Identifiant du sujet (ex: 'S001', 'S002', ..., 'S109').
    run_id : int
        Identifiant de la run (ex: 1, 2, ..., 14).
    Returns
    -------
    raw : mne.io.Raw
        Objet Raw contenant les données EEG chargées.
    """
    def load_subject_run(self, subject_id: str, run_id: int):
        """
        Charge une seule run (par fichier .edf) pour un sujet donné.
        """
        # Trouve le répertoire racine du projet (où se trouve src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/src
        project_root = os.path.dirname(script_dir)  # /path/to/project
        BASE_PATH = os.path.join(project_root, "src", "data", "MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
        
        filename = f"{subject_id}R{run_id:02}.edf"
        file_path = os.path.join(BASE_PATH, subject_id, filename)

        if not os.path.exists(file_path):
            subj_num = int(subject_id[1:])  # 'S014' -> 14
            print(f"📥 Téléchargement du sujet {subject_id}, run {run_id}...")
            mne.set_config("MNE_DATA", os.path.join(project_root, "data"))
            eegbci.load_data(subj_num, [run_id])

        raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', verbose=False)
        return raw

    """
    Extract epochs for a given run from raw data.
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    run : int
        Run identifier.
    Returns
    -------
    X : np.ndarray
        Extracted epochs data
    y : list
        Corresponding labels.
    """
    def extract_epochs_for_run(self, raw, run):
        events, event_id = mne.events_from_annotations(raw)

        # On garde uniquement T1 et T2
        filtered_event_id = {k: v for k, v in event_id.items() if k in ["T1", "T2"]}

        if not filtered_event_id:
            return [], []

        epochs = mne.Epochs(
            raw, events,
            event_id=filtered_event_id,
            tmin=-1.0, tmax=4.0, 
            baseline=None,
            preload=True,
            on_missing='ignore'
        )

        X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        
        if len(X) == 0:
            return [], []
        
        y = []
        for e in epochs.events:
            code = e[2]
            for k, v in filtered_event_id.items():
                if v == code:
                    y.append(useful_runs[run][k])
        
        if len(set(y)) < 2:
            print(f"Run {run} ignoré : seulement une classe présente.")
            return [], []
        
        return X, y
    
    """
    Setup data for all subjects and runs or for a specific subject and run.
    Parameters
    ----------
    subject_id : int, optional
        Subject identifier (1 to 109). If None, all subjects are processed.
    run_id : int, optional
        Run identifier. If None, all runs are processed.
    Returns
    -------
    subject : dict
        Dictionary containing the processed data.
    """
    def setup_data (self, subject_id=None, run_id=None):
        subject = {}
        if subject_id is None and run_id is None:
            # Charger TOUS les sujets et TOUTES les runs
            for i in range(109):
                subj_str = f"S{i + 1:03}"
                subject[subj_str] = {}
                for run, _ in useful_runs.items():
                    try:
                        raw_data = self.load_subject_run(subj_str, run)
                        raw_data = raw_data.filter(7., 30., fir_design='firwin')
                        subject[subj_str][run] = self.extract_epochs_for_run(raw_data, run)
                    except Exception as e:
                        print(f"Erreur pour {subj_str} run {run} : {e}")
        elif subject_id is None and run_id is not None:
            # Charger TOUS les sujets et UNE run spécifique
            for i in range(109):
                subj_str = f"S{i + 1:03}"
                subject[subj_str] = {}
                try:
                    raw_data = self.load_subject_run(subj_str, run_id)
                    raw_data = raw_data.filter(7., 30., fir_design='firwin')
                    subject[subj_str][run_id] = self.extract_epochs_for_run(raw_data, run_id)
                except Exception as e:
                    print(f"Erreur pour {subj_str} run {run_id} : {e}")
        elif subject_id is not None and run_id is None:
            # Charger UN sujet et TOUTES ses runs
            subj_str = f"S{subject_id:03}"
            subject[subj_str] = {}
            for run, _ in useful_runs.items():
                try:
                    raw_data = self.load_subject_run(subj_str, run)
                    raw_data = raw_data.filter(7., 30., fir_design='firwin')
                    subject[subj_str][run] = self.extract_epochs_for_run(raw_data, run)
                except Exception as e:
                    print(f"Erreur pour {subj_str} run {run} : {e}")
        else:
            # Charger UN sujet et UNE run spécifique
            subj_str = f"S{subject_id:03}"
            subject[subj_str] = {}
            try:
                raw_data = self.load_subject_run(subj_str, run_id)
                raw_data = raw_data.filter(7., 30., fir_design='firwin')
                subject[subj_str][run_id] = self.extract_epochs_for_run(raw_data, run_id)
            except Exception as e:
                print(f"Erreur pour {subj_str} run {run_id} : {e}")
        return subject
    
    def get_all_data(self, subject_id=None, run_id=None):
        """
        Get all data for specific subject(s) and run(s).
        
        Parameters
        ----------
        subject_id : int, optional
            Subject identifier (1 to 109). If None, all subjects are loaded.
        run_id : int, optional
            Run identifier. If None, all runs are loaded.
            
        Returns
        -------
        X_total : np.ndarray
            Extracted epochs data.
        y_total : np.ndarray
            Corresponding labels.
            
        Examples
        --------
        >>> p = Processing()
        >>> X, y = p.get_all_data()  # Tous les sujets, toutes les runs
        >>> X, y = p.get_all_data(4, 14)  # Sujet 4, run 14 seulement
        """
        subject = self.setup_data(subject_id, run_id)
        
        X_total = []
        y_total = []
        
        # Collecter toutes les epochs et trouver la taille temporelle minimale
        all_epochs = []
        min_n_times = float('inf')
        max_n_times = 0
        
        for subj_id in subject:
            for r_id in subject[subj_id]:
                X, y = subject[subj_id][r_id]
                if len(X) > 0:
                    all_epochs.append((X, y))
                    min_n_times = min(min_n_times, X.shape[2])
                    max_n_times = max(max_n_times, X.shape[2])
        
        # Si aucune donnée, retourner vide
        if not all_epochs:
            return np.array([]), np.array([])
        
        # Définir la taille cible (on garde seulement les epochs >= 95% de max_n_times)
        # Cela élimine les epochs tronquées tout en gardant les variations mineures
        target_n_times = max_n_times
        min_acceptable = int(max_n_times * 0.95)  # Tolérance de 5%
        
        # Filtrer et garder seulement les epochs complètes
        n_total = 0
        n_filtered = 0
        
        for X, y in all_epochs:
            for i, epoch in enumerate(X):
                n_total += 1
                epoch_len = epoch.shape[1]
                
                if epoch_len >= min_acceptable:
                    # Epoch assez longue, on la garde (crop au max si légèrement plus long)
                    X_total.append(epoch[:, :target_n_times])
                    y_total.append(y[i])
                else:
                    # Epoch trop courte, on l'ignore
                    n_filtered += 1
        
        if n_filtered > 0:
            print(f"ℹ️  Filtrage: {n_total} epochs → gardé {n_total - n_filtered} (rejeté {n_filtered} epochs trop courtes <{min_acceptable})")
        
        if len(X_total) == 0:
            return np.array([]), np.array([])

        X_total = np.array(X_total)
        y_total = np.array(y_total)
        
        print(f"✅ Chargé {len(X_total)} epochs de shape ({X_total.shape[1]} channels, {X_total.shape[2]} timepoints)")
        
        return X_total, y_total
    
    def get_combined_runs_data(self, subject_id, run_ids):
        """
        Charge et combine plusieurs runs pour un même sujet.
        Utile pour augmenter la quantité de données d'entraînement.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier (1 to 109).
        run_ids : list of int
            Liste des run IDs à combiner (ex: [3, 7, 11])
            
        Returns
        -------
        X_combined : np.ndarray
            Epochs combinées de tous les runs.
        y_combined : np.ndarray
            Labels correspondants.
        """
        X_all = []
        y_all = []
        
        for run_id in run_ids:
            X, y = self.get_all_data(subject_id, run_id)
            if len(X) > 0:
                X_all.append(X)
                y_all.extend(y)
        
        if not X_all:
            return np.array([]), np.array([])
        
        X_combined = np.concatenate(X_all, axis=0)
        y_combined = np.array(y_all)
        
        print(f"✅ Combiné {len(run_ids)} runs → {len(X_combined)} epochs totaux")
        
        return X_combined, y_combined
    
    def stream_epochs(self, subject_id, run_id):
        """
        Générateur qui yield les epochs un par un (vrai streaming).
        Simule le streaming en temps réel des données EEG.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier (1 to 109).
        run_id : int
            Run identifier.
            
        Yields
        ------
        epoch : np.ndarray
            Single epoch data (n_channels, n_times).
        label : str
            Corresponding label for this epoch.
        """
        subject = self.setup_data(subject_id, run_id)
        
        for subj_id in subject:
            for r_id in subject[subj_id]:
                X, y = subject[subj_id][r_id]
                # Yield epoch par epoch (streaming véritable)
                for epoch, label in zip(X, y):
                    yield epoch, label
