#!/usr/bin/env python3
"""Test rapide: Comparaison des stratégies d'entraînement."""

import numpy as np
from src.train_leave_one_run_out import train_leave_one_run_out
from src.train_incremental import train_subject_specific

print("="*80)
print("🔬 COMPARAISON: Stratégies d'Entraînement")
print("="*80)
print("\n1. Baseline: Train sur [3,7,11] → Test sur [4,8,12] mélangés")
print("2. Leave-One-Run-Out: Pour chaque test run, train sur tous les autres\n")

# Test sur 10 sujets
test_subjects = range(1, 11)

print("Test sur 10 sujets pour LEFT/RIGHT...\n")

# Approche 1: Baseline (actuelle)
print("📊 Baseline...")
results_baseline = train_subject_specific(
    subject_ids=test_subjects,
    calibration_runs=[3, 7, 11],
    test_runs=[4, 8, 12],
    task_name="left_right"
)

# Approche 2: Leave-One-Run-Out
print("\n📊 Leave-One-Run-Out...")
results_loro = train_leave_one_run_out(
    subject_ids=test_subjects,
    task_name="left_right"
)

# Comparaison
print("\n" + "="*80)
print("📊 COMPARAISON DES RÉSULTATS")
print("="*80)

if results_baseline and results_loro:
    baseline_mean = results_baseline['mean_test_score'] * 100
    loro_mean = results_loro['mean_test_score'] * 100
    
    print(f"\nBaseline:         {baseline_mean:.2f}%")
    print(f"Leave-One-Run-Out: {loro_mean:.2f}%")
    print(f"Gain:             {loro_mean - baseline_mean:+.2f}%")
    
    if loro_mean > baseline_mean:
        print(f"\n✅ Leave-One-Run-Out est MEILLEUR de {loro_mean - baseline_mean:.2f}%!")
    else:
        print(f"\n⚠️  Baseline reste meilleur")

print("\n" + "="*80 + "\n")
