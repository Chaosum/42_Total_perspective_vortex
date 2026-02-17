#!/usr/bin/env python3
"""
Script de monitoring pour suivre la progression de l'entraînement optimisé.
Affiche des statistiques en temps réel sur les résultats.
"""

import os
import time
import glob

def monitor_progress():
    """Monitore la progression de l'entraînement."""
    
    print("="*80)
    print("📊 MONITORING DE L'ENTRAÎNEMENT SUBJECT-SPECIFIC OPTIMISÉ")
    print("="*80)
    print("Surveillance des modèles créés dans models/subject_specific/\n")
    
    last_count = {'left_right': 0, 'hands_feet': 0}
    
    try:
        while True:
            print(f"\r[{time.strftime('%H:%M:%S')}]", end=" ")
            
            # Compter les modèles
            lr_models = glob.glob('models/subject_specific/left_right/*.pkl')
            hf_models = glob.glob('models/subject_specific/hands_feet/*.pkl')
            
            n_lr = len(lr_models)
            n_hf = len(hf_models)
            total = n_lr + n_hf
            expected = 109 * 2  # 109 sujets × 2 tâches
            
            # Calculer pourcentage
            pct = 100 * total / expected
            
            # Afficher avec barre de progression
            bar_length = 40
            filled = int(bar_length * total / expected)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"{bar} {pct:5.1f}% | LEFT/RIGHT: {n_lr:3d}/109 | HANDS/FEET: {n_hf:3d}/109", end="")
            
            # Nouveau modèle détecté?
            if n_lr > last_count['left_right']:
                print(f"\n   ✅ Nouveau modèle LEFT/RIGHT: {n_lr}/109", end="")
            if n_hf > last_count['hands_feet']:
                print(f"\n   ✅ Nouveau modèle HANDS/FEET: {n_hf}/109", end="")
            
            last_count = {'left_right': n_lr, 'hands_feet': n_hf}
            
            # Vérifier si terminé
            if total >= expected:
                print("\n\n🎉 ENTRAÎNEMENT TERMINÉ!")
                break
            
            time.sleep(5)  # Attendre 5 secondes
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring interrompu par l'utilisateur")
        print(f"Progression finale: {n_lr} + {n_hf} = {total}/{expected} modèles")

if __name__ == "__main__":
    monitor_progress()
