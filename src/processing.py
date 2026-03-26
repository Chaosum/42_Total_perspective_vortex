import os
import mne
import numpy as np
from mne.datasets import eegbci
from pathlib import Path
from autoreject import AutoReject
from utils import experiments

BASE_PATH = Path(__file__).parent.parent / "src" / "data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"

mne.set_log_level('WARNING')

def load_subject_run(subject_id: str, run_id: int):
	filename = f"{subject_id}R{run_id:02}.edf"
	file_path = BASE_PATH / subject_id / filename
	if not file_path.exists():
		subj_num = int(subject_id[1:])
		print(f"Téléchargement du sujet {subject_id}, run {run_id}...")
		mne.set_config("MNE_DATA", str(Path(__file__).parent.parent / "data"))
		eegbci.load_data(subj_num, [run_id])
	raw = mne.io.read_raw_edf(str(file_path), preload=True)
	return raw

def extract_epochs_for_run(raw, run):
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

def extract_epochs(raw):
	"""Extrait les epochs des annotations, ignorant Rest"""
	events, event_id = mne.events_from_annotations(raw)
	
	# Ignorer "Rest" et créer un nouveau event_id avec les autres classes
	filtered_event_id = {k: v for k, v in event_id.items() if "Rest" not in str(k) and "T0" not in str(k) and "0" not in str(k)}
	
	if len(filtered_event_id) < 2:
		# Si pas exactement 2 classes, retourner None
		return None
	
	epochs = mne.Epochs(
		raw, events,
		event_id=filtered_event_id,
		tmin=-1.0, tmax=4.0,
		baseline=None,
		preload=True,
		on_missing='ignore'
	)
	
	if len(epochs) == 0:
		return None
		
	return epochs

def clean_epochs_autoreject(epochs):
	"""Nettoyage léger: rejette les epochs avec amplitude anormale"""
	try:
		X = epochs.get_data()
		# Calculer l'amplitude max par epoch
		amplitudes = np.max(np.abs(X), axis=(1, 2))
		
		# Seuil: mean + 3*std
		threshold = np.mean(amplitudes) + 3 * np.std(amplitudes)
		
		# Masque des epochs à garder
		mask = amplitudes < threshold
		
		return epochs[mask]
	except Exception as e:
		return epochs

def balance_classes(epochs):
	"""Balance les classes"""
	min_count = np.inf
	for label in np.unique(epochs.events[:, -1]):
		count = np.sum(epochs.events[:, -1] == label)
		min_count = min(min_count, count)
	min_count = int(min_count)
	indices = []
	for label in np.unique(epochs.events[:, -1]):
		label_indices = np.where(epochs.events[:, -1] == label)[0]
		indices.extend(np.random.choice(label_indices, min_count, replace=False))
	return epochs[sorted(indices)]

def average_over_epochs(epochs, window_size=5, overlap=0.5):
	"""Crée des super-epochs en moyennant N epochs consécutifs avec chevauchement
	
	Args:
		window_size: nombre d'epochs à moyenner pour chaque super-epoch
		overlap: fraction de chevauchement entre fenêtres (0=none, 1=full)
			stride = window_size * (1 - overlap)
	"""
	labels = epochs.events[:, -1]
	X = epochs.get_data()
	X_avg = []
	y_avg = []
	
	event_id_inv = {v: k for k, v in epochs.event_id.items()}
	
	# Calculer le stride basé sur le chevauchement
	stride = max(1, int(window_size * (1 - overlap)))
	
	# Pour chaque classe, créer des super-epochs en moyennant window_size epochs
	for label in np.unique(labels):
		mask = np.where(labels == label)[0]
		class_data = X[mask]
		
		# Créer des super-epochs avec fenêtres chevauchantes
		for i in range(0, len(class_data) - window_size + 1, stride):
			super_epoch = np.mean(class_data[i:i+window_size], axis=0)
			X_avg.append(super_epoch)
			y_avg.append(event_id_inv[label])
	
	return np.array(X_avg), np.array(y_avg)

def split_train_test(X, y, test_size=0.2, random_state=42):
	"""Split data into train/test sets with fixed random state"""
	rng = np.random.RandomState(random_state)
	n_samples = len(X)
	n_test = int(n_samples * test_size)
	
	indices = np.arange(n_samples)
	rng.shuffle(indices)
	
	test_indices = indices[:n_test]
	train_indices = indices[n_test:]
	
	X_train, X_test = X[train_indices], X[test_indices]
	y_train, y_test = y[train_indices], y[test_indices]
	
	return X_train, X_test, y_train, y_test

def setup_data(subject_id, run_id):
	subject = {}
	subj_str = f"S{subject_id:03}"
	subject[subj_str] = {}
	try:
		raw_data = load_subject_run(subj_str, run_id)
		raw_data = raw_data.filter(7., 30., fir_design='firwin')
		subject[subj_str][run_id] = extract_epochs_for_run(raw_data, run_id)
	except Exception as e:
		print(f"Erreur pour {subj_str} run {run_id} : {e}")
	return subject

def get_data(subject_id, run_id):
	subject = setup_data(subject_id=subject_id, run_id=run_id)
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

def setup_all_data(experiment, max_subjects=None):
	raws = []
	max_range = min(110, max_subjects + 1) if max_subjects else 110
	for i in range(1, max_range):
		subj_str = f"S{i:03}"
		for run in experiment["runs"]:
			try:
				raw_data = load_subject_run(subj_str, run)
				if raw_data.info['sfreq'] != 160.0:
					raw_data.resample(sfreq=160.0)
				mne.datasets.eegbci.standardize(raw_data)
				raw_data.set_montage("standard_1005")
				events, _ = mne.events_from_annotations(raw_data)
				# Utiliser le mapping de l'expérience fournie, pas chercher ailleurs
				mapping = experiment["mapping"]
				annotations = mne.annotations_from_events(
					events=events,
					event_desc=mapping,
					sfreq=raw_data.info["sfreq"]
				)
				raw_data.set_annotations(annotations)
				raws.append(raw_data)
			except Exception as e:
				print(f"Erreur pour {subj_str} run {run} : {e}")
	
	if len(raws) == 0:
		return None
	
	raw = raws[0]
	for r in raws[1:]:
		raw.append(r)
	return raw

def setup_data_for_subject(experiment, subject_id):
	"""Charge tous les runs d'UN sujet pour une expérience donnée"""
	raws = []
	subj_str = f"S{subject_id:03}"
	for run in experiment["runs"]:
		try:
			raw_data = load_subject_run(subj_str, run)
			if raw_data.info['sfreq'] != 160.0:
				raw_data.resample(sfreq=160.0)
			mne.datasets.eegbci.standardize(raw_data)
			raw_data.set_montage("standard_1005")
			events, _ = mne.events_from_annotations(raw_data)
			mapping = experiment["mapping"]
			annotations = mne.annotations_from_events(
				events=events,
				event_desc=mapping,
				sfreq=raw_data.info["sfreq"]
			)
			raw_data.set_annotations(annotations)
			raws.append(raw_data)
		except Exception as e:
			print(f"[ERROR] {subj_str} run {run}: {e}")
			return None
	
	if len(raws) == 0:
		return None
	
	raw = raws[0]
	for r in raws[1:]:
		raw.append(r)
	return raw
		
def get_all_data(experiment, max_subjects=None):
	return setup_all_data(experiment, max_subjects)