# Comparaison Finale des Modèles

Le gain des Transformers n'est **pas mesurable** sur ce jeu de données spécifique, car le modèle classique (Logistic Regression) obtient déjà un F1-macro parfait de 1.0.
Ce résultat met en évidence une limitation majeure : le jeu de données (1250 CVs) est trop simple, sans ambiguïté ou potentiellement fuité, rendant le test set non représentatif de la complexité réelle.
Déployer DistilBERT coûterait 1420s d'entraînement et ~65ms d'inférence (taille > 250MB), pour **zéro** gain de performance.
Le choix de déploiement idéal est donc **Logistic Regression (TF-IDF)** pour son inférence instantanée (<1ms) et sa légèreté (<1MB).
Test Statistique (McNemar) : Les prédictions étant identiques, la p-value est de 1.0 (aucune différence significative).
> **Note sur l'API Flask :** Le code actuel charge bien le meilleur modèle classique, mais le nom affiché est codé en dur comme 'Linear SVM (TF-IDF)' au lieu du modèle réel (Logistic Regression). Il charge aussi SBERT en parallèle pour l'analyse duale.
