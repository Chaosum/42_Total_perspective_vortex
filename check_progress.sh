#!/bin/bash
# Quick progress check for CSP optimization training

echo "=========================================="
echo "📊 CSP OPTIMIZATION - PROGRESS CHECK"
echo "=========================================="

# Count completed subjects
n_optimized=$(grep "✅ Optimal" train_subject_specific_optimized.log 2>/dev/null | wc -l)
echo "Sujets optimisés: $n_optimized/218 ($(echo "scale=1; $n_optimized*100/218" | bc)%)"

# Count models saved
n_lr=$(ls models/subject_specific/left_right/*.pkl 2>/dev/null | wc -l)
n_hf=$(ls models/subject_specific/hands_feet/*.pkl 2>/dev/null | wc -l)
echo "Modèles LEFT/RIGHT: $n_lr/109"
echo "Modèles HANDS/FEET: $n_hf/109"

# Distribution of n_components
echo ""
echo "Distribution des composantes CSP:"
grep "✅ Optimal: n_components=" train_subject_specific_optimized.log 2>/dev/null | \
    sed 's/.*n_components=\([0-9]\).*/\1/' | \
    sort | uniq -c | \
    awk '{printf "  %d composantes: %3d sujets\n", $2, $1}'

# Best scores
echo ""
echo "Top 5 scores de test:"
grep "Score test:" train_subject_specific_optimized.log 2>/dev/null | \
    awk '{print $3}' | \
    sort -rn | \
    head -5 | \
    awk '{printf "  %.4f (%.1f%%)\n", $1, $1*100}'

# Mean score so far
echo ""
echo "Mean score actuel:"
grep "Score test:" train_subject_specific_optimized.log 2>/dev/null | \
    awk '{sum+=$3; n++} END {if(n>0) printf "  %.4f (%.1f%%)\n", sum/n, sum/n*100; else print "  Pas encore de données"}'

echo ""
echo "=========================================="
echo "⏱️  Pour suivre en temps réel:"
echo "   tail -f train_subject_specific_optimized.log"
echo "=========================================="
