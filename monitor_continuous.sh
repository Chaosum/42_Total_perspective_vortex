#!/bin/bash
# Script de monitoring continu avec mise à jour automatique

echo "=========================================="
echo "🔄 MONITORING CONTINU - CSP OPTIMIZATION"
echo "=========================================="
echo "Appuyez sur Ctrl+C pour arrêter"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "📊 CSP OPTIMIZATION - $(date '+%H:%M:%S')"
    echo "=========================================="
    
    # Progression
    n_optimized=$(grep "✅ Optimal" train_subject_specific_optimized.log 2>/dev/null | wc -l)
    pct=$(echo "scale=1; $n_optimized*100/218" | bc)
    
    # Barre de progression
    filled=$(echo "scale=0; $n_optimized*40/218" | bc)
    bar=""
    for i in $(seq 1 $filled); do bar="${bar}█"; done
    for i in $(seq $filled 40); do bar="${bar}░"; done
    
    echo "Progression: $bar $pct%"
    echo "Sujets: $n_optimized/218"
    echo ""
    
    # Distribution CSP
    echo "Distribution des composantes CSP:"
    grep "✅ Optimal: n_components=" train_subject_specific_optimized.log 2>/dev/null | \
        sed 's/.*n_components=\([0-9]\).*/\1/' | \
        sort | uniq -c | \
        awk '{printf "  %d composantes: %3d sujets (%.1f%%)\n", $2, $1, $1*100/'$n_optimized'}'
    
    echo ""
    
    # Statistiques de performance
    mean_score=$(grep "Score test:" train_subject_specific_optimized.log 2>/dev/null | \
        awk '{sum+=$3; n++} END {if(n>0) printf "%.2f", sum/n*100}')
    
    max_score=$(grep "Score test:" train_subject_specific_optimized.log 2>/dev/null | \
        awk '{print $3}' | sort -rn | head -1 | awk '{printf "%.2f", $1*100}')
    
    echo "Performance actuelle:"
    echo "  Mean accuracy: ${mean_score}%"
    echo "  Best subject:  ${max_score}%"
    
    # Estimation temps restant
    if [ $n_optimized -gt 0 ]; then
        # Calculer le temps écoulé depuis le début
        start_time=$(stat -c %Y train_subject_specific_optimized.log 2>/dev/null)
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        # Temps moyen par sujet
        avg_time=$((elapsed / n_optimized))
        
        # Temps restant estimé
        remaining=$((avg_time * (218 - n_optimized)))
        hours=$((remaining / 3600))
        minutes=$(((remaining % 3600) / 60))
        
        echo ""
        echo "Temps restant estimé: ${hours}h ${minutes}min"
    fi
    
    echo ""
    echo "=========================================="
    echo "Mise à jour dans 10 secondes..."
    echo "=========================================="
    
    sleep 10
done
