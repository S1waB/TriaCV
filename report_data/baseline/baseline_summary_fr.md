# Résumé Baseline (Modèles Classiques)

Le meilleur modèle classique est **Logistic Regression** avec un F1-macro de 1.0000 et une précision macro de 1.0000 sur l'ensemble de test.
La validation croisée a confirmé sa robustesse avec un F1-macro moyen de 1.0000.
L'écart de performance entre le meilleur (Logistic Regression) et le moins performant (Multinomial Naive Bayes) est de 0.0000 (F1-macro).
En termes de compromis performance/temps, Logistic Regression s'entraîne en 7.0 secondes.
L'inférence est extrêmement rapide avec environ 0.01 millisecondes par document, ce qui est idéal pour une API temps réel.
