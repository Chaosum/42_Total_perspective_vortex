import argparse
import os
import sys
import joblib
import numpy as np
import time
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, ShuffleSplit
from tqdm import tqdm

# Fixer le random seed pour reproductibilité
np.random.seed(42)

# Supprimer les avertissements MNE bénins
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*annotation.*")
from processing import get_all_data, get_data, extract_epochs, balance_classes, average_over_epochs, split_train_test
from utils import experiments
from MyCSP import MyCSP
from sklearn.pipeline import Pipeline

# Accès aux modules du dossier src/ quel que soit le répertoire de lancement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Chemin racine des modèles (relatif au répertoire de lancement)
def _model_path(subject_id: int, task: str, test_run: int) -> str:
	base = os.path.join(
		os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
		"models", "leave_one_run_out", task
	)
	return os.path.join(base, f"subject_{subject_id:03d}_testrun_{test_run}.pkl")

def main():
	parser = argparse.ArgumentParser(
		description="",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=(
			"Exemples :\n"
			"  python mybci.py\n"
			"  python mybci.py <subject_id 1 - 109> <run_id 3 - 14> train\n"
			"  python mybci.py <subject_id 1 - 109> <run_id 3 - 14> predict\n"
		),
	)

	parser.add_argument("subject_id", type=int, nargs="?", default=None,
						help="Numéro du sujet (1–109)")
	parser.add_argument("run_id", type=int, nargs="?", default=None,
						help="Numéro du run (3–14)")
	parser.add_argument("command", nargs="?", default=None,
						choices=["train", "predict"],
						help="Commande à exécuter")

	args = parser.parse_args()

	if args.command is None:
		runAlltests()

	if args.command == "train":
		train(subject_id=args.subject_id, run_id=args.run_id)
	elif args.command == "predict":
		predict(subject_id=args.subject_id, run_id=args.run_id)


def print_results(exps):
	"""Affiche les scores train/test/crossval pour chaque expérience"""
	train_scores = []
	test_scores = []
	cv_mean_scores = []
	
	for experiment in exps:
		if "pipeline" in experiment and "X_train" in experiment:
			train_score = experiment["pipeline"].score(experiment["X_train"], experiment["y_train"])
			test_score = experiment["pipeline"].score(experiment["X_test"], experiment["y_test"])
			cv_mean = np.mean(experiment["cv_scores"])
			
			train_scores.append(train_score)
			test_scores.append(test_score)
			cv_mean_scores.append(cv_mean)
			
			print(f"\n{experiment['name']}")
			print(f"  Train:    {train_score:.3f}")
			print(f"  Test:     {test_score:.3f}")
			print(f"  CV:       {cv_mean:.3f} (+/- {np.std(experiment['cv_scores']):.3f})")
	
	if len(train_scores) == 0:
		print("\n❌ Aucun résultat à afficher - vérifier les logs d'erreur")
		return
		
	print("\n" + "="*50)
	print("=== Moyennes globales ===")
	print(f"Train:      {np.mean(train_scores):.3f} +/- {np.std(train_scores):.3f}")
	print(f"Test:       {np.mean(test_scores):.3f} +/- {np.std(test_scores):.3f}")
	print(f"CV:         {np.mean(cv_mean_scores):.3f} +/- {np.std(cv_mean_scores):.3f}")


def runAlltests():
	print("\n=== Démarrage du traitement ===\n")
	start_total = time.time()
	
	for i, experiment in enumerate(tqdm(experiments, desc="Experiments")):
		print(f"\n[{i+1}/{len(experiments)}] Traitement de {experiment['name']}...")
		
		# Charger données
		t0 = time.time()
		raw = get_all_data(experiment)
		if raw is None:
			print(f"  ✗ Pas de données")
			continue
		print(f"  ✓ Données chargées ({time.time()-t0:.1f}s)")
		
		# Filtrer
		t0 = time.time()
		raw = raw.notch_filter(60, method="iir")
		raw = raw.filter(1., 15., fir_design='firwin', skip_by_annotation="edge")
		print(f"  ✓ Filtrage appliqué ({time.time()-t0:.1f}s)")
		
		# Extraire epochs
		t0 = time.time()
		epochs = extract_epochs(raw)
		if epochs is None:
			print(f"  ✗ Pas d'epochs")
			continue
		print(f"  ✓ Epochs extraits ({time.time()-t0:.1f}s)")
		
		# Balancer et moyenner
		t0 = time.time()
		epochs = balance_classes(epochs)
		X_avg, y_avg = average_over_epochs(epochs)
		print(f"  ✓ Classes balancées et moyennées ({time.time()-t0:.1f}s) - {len(X_avg)} échantillons")
		
		experiment["epochs"] = epochs
		experiment["X_avg"] = X_avg
		experiment["y_avg"] = y_avg

		# Train/Test split
		X_train, X_test, y_train, y_test = split_train_test(X_avg, y_avg)
		experiment["X_train"] = X_train
		experiment["X_test"] = X_test
		experiment["y_train"] = y_train
		experiment["y_test"] = y_test

		# Entraîner
		t0 = time.time()
		csp = MyCSP(n_components=4)
		lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')
		pipeline = Pipeline([
			("CSP", csp),
			("LDA", lda)
		])
		pipeline.fit(X_train, y_train)
		experiment["pipeline"] = pipeline
		joblib.dump(
			pipeline,
			f'{experiment["name"]}.joblib'
		)
		print(f"  ✓ Pipeline entraîné et sauvegardé ({time.time()-t0:.1f}s)")
		
		# Cross-validation avec ShuffleSplit (comme le repo original)
		t0 = time.time()
		cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
		cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
		experiment["cv_scores"] = cv_scores
		print(f"  ✓ Cross-validation complétée ({time.time()-t0:.1f}s)")
	
	elapsed = time.time() - start_total
	print(f"\n✓ Traitement terminé en {elapsed/60:.1f} minutes")
	print_results(experiments)


def train(subject_id: int, run_id: int):
	"""Entraîne un modèle LORO pour un sujet donné"""
	print(f"\n=== Entraînement sujet {subject_id}, run {run_id} ===\n")
	
	# Utiliser la première expérience (assumption: une seule task par défaut)
	experiment = experiments[0]
	
	# Charger et traiter les données pour ce sujet/run
	raw = get_data(subject_id, run_id)
	if raw is None or len(raw[0]) == 0:
		print(f"Pas de données pour sujet {subject_id}, run {run_id}")
		return
	
	X, y = raw
	X_train, X_test, y_train, y_test = split_train_test(X, y)
	
	# Créer et entraîner le pipeline
	csp = MyCSP(n_components=4)
	lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')
	pipeline = Pipeline([
		("CSP", csp),
		("LDA", lda)
	])
	pipeline.fit(X_train, y_train)
	
	# Évaluer
	train_score = pipeline.score(X_train, y_train)
	test_score = pipeline.score(X_test, y_test)
	
	print(f"Train: {train_score:.2f}")
	print(f"Test:  {test_score:.2f}")
	
	# Sauvegarder
	model_path = _model_path(subject_id, experiment["task"] if "task" in experiment else "default", run_id)
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	joblib.dump(pipeline, model_path)
	print(f"Modèle sauvegardé: {model_path}")


def predict(subject_id: int, run_id: int):
	"""Prédit avec un modèle entraîné pour un sujet donné"""
	print(f"\n=== Prédiction sujet {subject_id}, run {run_id} ===\n")
	
	experiment = experiments[0]
	model_path = _model_path(subject_id, experiment["task"] if "task" in experiment else "default", run_id)
	
	# Charger le modèle
	try:
		pipeline = joblib.load(model_path)
	except FileNotFoundError:
		print(f"Modèle non trouvé: {model_path}")
		return
	
	# Charger et traiter les données
	raw = get_data(subject_id, run_id)
	if raw is None or len(raw[0]) == 0:
		print(f"Pas de données pour sujet {subject_id}, run {run_id}")
		return
	
	X, y = raw
	predictions = pipeline.predict(X)
	accuracy = np.mean(predictions == y)
	
	print(f"Accuracy: {accuracy:.2f}")
	print(f"Prédictions: {predictions[:10]}... (premières 10)")
	print(f"Vraies valeurs: {y[:10]}... (premières 10)")


if __name__ == "__main__":
	main()

