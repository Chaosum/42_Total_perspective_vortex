
import os
import mne
import numpy as np
from mne.datasets import eegbci

mne.set_log_level('WARNING')

class Processing:
	def __init__(self):
		self._data = {}

	def load_subject_run(self, subject_id: str, run_id: int):
		script_dir = os.path.dirname(os.path.abspath(__file__))
		project_root = os.path.dirname(script_dir)
		BASE_PATH = os.path.join(project_root, "src", "data", "MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
		filename = f"{subject_id}R{run_id:02}.edf"
		file_path = os.path.join(BASE_PATH, subject_id, filename)
		if not os.path.exists(file_path):
			subj_num = int(subject_id[1:])
			print(f"Téléchargement du sujet {subject_id}, run {run_id}...")
			mne.set_config("MNE_DATA", os.path.join(project_root, "data"))
			eegbci.load_data(subj_num, [run_id])
		raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', verbose=False)
		return raw

	def extract_epochs_for_run(self, raw, run):
		events, event_id = mne.events_from_annotations(raw)
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
		X = epochs.get_data()
		if len(X) == 0:
			return [], []
		y = []
		for e in epochs.events:
			code = e[2]
			for k, v in filtered_event_id.items():
				if v == code:
					y.append(k)
		if len(set(y)) < 2:
			print(f"Run {run} ignoré : seulement une classe présente.")
			return [], []
		return X, y

	def setup_data(self, subject_id=None, run_id=None):
		subject = {}
		if subject_id is not None and run_id is not None:
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
		subject = self.setup_data(subject_id, run_id)
		X_total = []
		y_total = []
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
		if not all_epochs:
			return np.array([]), np.array([])
		target_n_times = max_n_times
		min_acceptable = int(max_n_times * 0.95)
		n_total = 0
		n_filtered = 0
		for X, y in all_epochs:
			for i, epoch in enumerate(X):
				n_total += 1
				epoch_len = epoch.shape[1]
				if epoch_len >= min_acceptable:
					X_total.append(epoch[:, :target_n_times])
					y_total.append(y[i])
				else:
					n_filtered += 1
		if len(X_total) == 0:
			return np.array([]), np.array([])
		X_total = np.array(X_total)
		y_total = np.array(y_total)
		return X_total, y_total
