# Résumé Factual de l'EDA

Le jeu de données contient 1250 CVs répartis en 25 catégories (ratio de déséquilibre de 1.0). On observe 0 quasi-doublons, et la déduplication n'est pas appliquée avant le split d'entraînement. La longueur médiane des CVs est de 258.0 mots, mais 0.0% des documents dépassent la limite de 512 tokens de DistilBERT. Le prétraitement réduit significativement le vocabulaire de 1169 à 648 termes uniques. Les données ne présentent aucune valeur manquante, validant leur qualité pour l'apprentissage automatique.
