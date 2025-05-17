import sys
import os
import numpy as np
import joblib
from processing import Processing
from global_variable import useful_runs

def eval_subject_run(subject_id, run_id, wavelet_pipeline, csp_pipeline, clf):
    subject_str = f"S{subject_id:03}"
    p = Processing()
    data = p.setup_data([subject_str], [run_id])

    X_total = []
    y_total = []
    for run in data[subject_str]:
        X, y = data[subject_str][run]
        X_total.extend(X)
        y_total.extend(y)

    if len(set(y_total)) < 2:
        return None  # skip mono-class runs

    X_total = np.array(X_total)
    y_total = np.array(y_total)

    X_csp = csp_pipeline.transform(X_total)
    X_wavelet = wavelet_pipeline.transform(X_total)
    X_combined = np.hstack([X_csp, X_wavelet])

    acc = clf.score(X_combined, y_total)
    return acc

if __name__ == "__main__":
    # === Mode 1 : sans argument → test global sur 6 expériences ===
    if len(sys.argv) == 1:
        main_experiments = [4, 6, 7, 8, 9, 10]  # runs choisis
        wavelet_pipeline = joblib.load("wavelet_pipeline.pkl")
        csp_pipeline = joblib.load("csp_pipeline.pkl")
        clf = joblib.load("classifier.pkl")

        experiment_accuracies = {}
        for i, run_id in enumerate(main_experiments):
            run_scores = []
            for subject_id in range(1, 110):
                acc = eval_subject_run(subject_id, run_id, wavelet_pipeline, csp_pipeline, clf)
                if acc is not None:
                    run_scores.append(acc)
                    print(f"experiment {i}: subject {subject_id:03}: accuracy = {acc:.4f}")

            mean_acc = np.mean(run_scores)
            experiment_accuracies[i] = mean_acc
            print(f"\nMean accuracy of experiment {i}: {mean_acc:.4f}\n")

        print("📊 Mean accuracy of 6 experiments:")
        for i in experiment_accuracies:
            print(f"experiment {i}: accuracy = {experiment_accuracies[i]:.4f}")
        print(f"\n🔚 Final mean accuracy: {np.mean(list(experiment_accuracies.values())):.4f}")
        sys.exit(0)

    # === Mode 2 : avec arguments → subject, run, mode ===
    if len(sys.argv) != 4:
        print("Usage: python mybci.py <subject_num> <run_num> <train/predict>")
        sys.exit(1)

    subject = int(sys.argv[1])
    run = int(sys.argv[2])
    mode = sys.argv[3]

    if not (1 <= subject <= 109):
        print("❌ Subject must be between 1 and 109 (S001 to S109)")
        sys.exit(1)

    if run not in useful_runs:
        print(f"❌ Run must be one of: {list(useful_runs.keys())}")
        sys.exit(1)

    os.environ["BCI_SUBJECT"] = f"S{subject:03}"
    os.environ["BCI_RUN"] = str(run)

    if mode == "train":
        os.system("python train.py")
    elif mode == "predict":
        os.system("python predict.py")
    else:
        print("❌ Mode must be 'train' or 'predict'")
        sys.exit(1)
