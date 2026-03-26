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
from processing import setup_data_for_subject, get_data, extract_epochs, balance_classes, average_over_epochs, split_train_test, clean_epochs_autoreject
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
			"  python mybci.py --per-subject [1-6]\n"
			"  python mybci.py <subject_id 1 - 109> <run_id 3 - 14> train\n"
			"  python mybci.py <subject_id 1 - 109> <run_id 3 - 14> predict\n"
		),
	)
	
	parser.add_argument("--per-subject", type=int, default=None, choices=[1,2,3,4,5,6],
						help="Entraîner UN modèle par sujet pour une expérience donnée (1-6)")
	parser.add_argument("subject_id", type=int, nargs="?", default=None,
						help="Numéro du sujet (1–109)")
	parser.add_argument("run_id", type=int, nargs="?", default=None,
						help="Numéro du run (3–14)")
	parser.add_argument("command", nargs="?", default=None,
						choices=["train", "predict"],
						help="Commande à exécuter")

	args = parser.parse_args()

	if args.per_subject:
		run_per_subject(args.per_subject)
	elif args.command is None:
		runAlltests()
	elif args.command == "train":
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


def run_per_subject(experiment_id=1):
	"""Entraîne UN modèle PAR SUJET pour une expérience donnée"""
	print(f"\n=== Entraînement par sujet: {experiments[experiment_id-1]['name']} ===\n")
	experiment = experiments[experiment_id-1]
	results = []
	
	for subject_id in tqdm(range(1, 110), desc=f"Sujets"):
		try:
			# Charger données
			raw = setup_data_for_subject(experiment, subject_id)
			if raw is None:
				if subject_id <= 2:
					print(f"[*] S{subject_id}: raw is None")
				continue
			
			# Filtrer
			raw = raw.notch_filter(60, method="iir")
			raw = raw.filter(8., 40., fir_design='firwin', skip_by_annotation="edge")
			
			# Extraire epochs
			epochs = extract_epochs(raw)
			if epochs is None:
				if subject_id <= 2:
					print(f"[*] S{subject_id}: epochs is None")
				continue
			
			# Nettoyer + balancer + moyenner
			epochs = clean_epochs_autoreject(epochs)
			epochs = balance_classes(epochs)
			# For per-subject: use smaller window size with overlap to get enough super-epochs
			X_avg, y_avg = average_over_epochs(epochs, window_size=5, overlap=0.5)
			
			if len(X_avg) < 10:
				if subject_id <= 2:
					print(f"[*] S{subject_id}: len(X_avg)={len(X_avg)} < 10")
				continue
			
			# Train/Test split
			X_train, X_test, y_train, y_test = split_train_test(X_avg, y_avg)
			
			# Entraîner
			csp = MyCSP(n_components=4)
			lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')
			pipeline = Pipeline([("CSP", csp), ("LDA", lda)])
			pipeline.fit(X_train, y_train)
			
			# Scorer
			train_score = pipeline.score(X_train, y_train)
			test_score = pipeline.score(X_test, y_test)
			
			# Cross-validation
			cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
			cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
			
			results.append({
				'subject': subject_id,
				'train': train_score,
				'test': test_score,
				'cv': np.mean(cv_scores)
			})
		except Exception as e:
			print(f"\n[ERR] Subject {subject_id} exception: {type(e).__name__}: {str(e)[:80]}")
	
	# Afficher résultats
	if results:
		train_scores = [r['train'] for r in results]
		test_scores = [r['test'] for r in results]
		cv_scores = [r['cv'] for r in results]
		
		print(f"\n[OK] {len(results)} sujets traites")
		print("\n=== Moyennes par sujet ===")
		print(f"Train:  {np.mean(train_scores):.3f} +/- {np.std(train_scores):.3f}")
		print(f"Test:   {np.mean(test_scores):.3f} +/- {np.std(test_scores):.3f}")
		print(f"CV:     {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")
		
		# Top 10
		results_sorted = sorted(results, key=lambda x: x['test'], reverse=True)
		print(f"\n[TOP5] Top 5 sujets (par test accuracy):")
		for i, r in enumerate(results_sorted[:5]):
			print(f"  {i+1}. Sujet {r['subject']}: test={r['test']:.3f}, train={r['train']:.3f}, cv={r['cv']:.3f}")
	else:
		print("[ERR] Aucun sujet traité avec succès")
	
	# Retourner les résultats pour utilisation dans visualize_results.py
	return results


def runAlltests():
	print("\n=== Démarrage du traitement ===\n")
	start_total = time.time()
	
	# Accumulator pour les scores moyens par expérience
	mean_scores = {i: [] for i in range(len(experiments))}
	
	# Boucler sur chaque expérience
	for exp_idx, experiment in enumerate(experiments):
		print(f"[DEBUG] Exp {exp_idx}: {experiment['name']}")
		# Boucler sur chaque sujet (1-109)
		count_subjects = 0
		for subject_id in range(1, 110):
			try:
				# Charger les données du sujet pour cette expérience
				raw = setup_data_for_subject(experiment, subject_id)
				if raw is None:
					continue
				
				# Filtrer
				raw = raw.notch_filter(60, method="iir")
				raw = raw.filter(1., 15., fir_design='firwin', skip_by_annotation="edge")
				
				# Extraire epochs
				epochs = extract_epochs(raw)
				if epochs is None:
					continue
				
				# Nettoyer
				epochs = clean_epochs_autoreject(epochs)
				
				# Balancer et moyenner
				epochs = balance_classes(epochs)
				X_avg, y_avg = average_over_epochs(epochs)
				
				if len(X_avg) == 0:
					continue
				
				# Train/Test split
				X_train, X_test, y_train, y_test = split_train_test(X_avg, y_avg)
				
				# Entraîner
				csp = MyCSP(n_components=4)
				lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto')
				pipeline = Pipeline([
					("CSP", csp),
					("LDA", lda)
				])
				pipeline.fit(X_train, y_train)
				
				# Évaluer sur test set
				test_score = pipeline.score(X_test, y_test)
				
				# Cross-validation sur le train set
				cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
				cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
				cv_mean = np.nanmean(cv_scores)
				
				mean_scores[exp_idx].append(test_score)
				count_subjects += 1
				
				# Afficher le résultat
				print(f"experiment {exp_idx}: subject {subject_id:03d}: accuracy = {test_score:.4f}")
			
			except Exception as e:
				# Afficher l'erreur pour debug
				print(f"[ERROR] S{subject_id:03} exp{exp_idx}: {type(e).__name__}: {str(e)[:80]}")
		
		print(f"[DEBUG] Exp {exp_idx}: {count_subjects} subjects processed")
	
	# Afficher les moyennes
	print(f"\nMean accuracy of the six different experiments for all 109 subjects:")
	for exp_idx in range(len(experiments)):
		scores = mean_scores[exp_idx]
		if scores:
			mean_acc = np.mean(scores)
			print(f"experiment {exp_idx}: accuracy = {mean_acc:.4f}")
	
	elapsed = time.time() - start_total
	print(f"\n[OK] Traitement termine en {elapsed/60:.1f} minutes")



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
	
	# Évaluer avec sklearn scoring tools
	train_score = pipeline.score(X_train, y_train)
	test_score = pipeline.score(X_test, y_test)
	cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
	
	print(f"Train: {train_score:.4f}, Test: {test_score:.4f} (gap={train_score-test_score:.4f})")
	print("CV scores:")
	for i, score in enumerate(cv_scores, 1):
		print(f"  Fold {i}: {score:.4f}")
	print(f"cross_val_score: {cv_scores.mean():.4f}")
	
	# Sauvegarder
	model_path = _model_path(subject_id, experiment["task"] if "task" in experiment else "default", run_id)
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	joblib.dump(pipeline, model_path)
	print(f"Modèle sauvegardé: {model_path}")


def predict(subject_id: int, run_id: int):
	"""Mode streaming: traite les epochs au fur et à mesure et affiche les prédictions en temps réel"""
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
	
	# Traiter les epochs un par un (streaming)
	correct = 0
	predictions = []
	
	print(f"Traitement de {len(X)} epochs en streaming...\n")
	start_time = time.time()
	
	for i, (epoch, true_label) in enumerate(zip(X, y)):
		# Prédire sur cet epoch individuel
		epoch_reshaped = epoch.reshape(1, *epoch.shape)  # Ajouter dimension batch
		pred = pipeline.predict(epoch_reshaped)[0]
		predictions.append(pred)
		
		# Vérifier si correct
		is_correct = (pred == true_label)
		correct += is_correct
		
		# Affichage en temps réel (chaque 5 epochs ou le dernier)
		if (i + 1) % 5 == 0 or i == len(X) - 1:
			elapsed = time.time() - start_time
			accuracy_so_far = correct / (i + 1)
			print(f"[{elapsed:.2f}s] Epoch {i+1}/{len(X)}: Prédiction={pred}, Vrai={true_label}, Accuracy=({correct}/{i+1}={accuracy_so_far:.2%})")
	
	# Résultats finaux
	total_time = time.time() - start_time
	final_accuracy = np.mean(np.array(predictions) == y)
	
	print(f"\n=== Résultats ===")
	print(f"Temps total: {total_time:.2f}s")
	print(f"Temps moyen par epoch: {total_time/len(X)*1000:.1f}ms")
	print(f"Accuracy finale: {final_accuracy:.2%}")
	print(f"Prédictions: {predictions[:10]}... (premières 10)")
	print(f"Vraies valeurs: {y[:10]}... (premières 10)")


if __name__ == "__main__":
	main()

