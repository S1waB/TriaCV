# Résumé de l'Application

Le flux de l'application commence par l'upload d'un CV (fichier ou texte) depuis l'interface React, qui est validé puis envoyé à l'API Flask.
L'API procède à l'extraction du texte brut (PDF/Docx via libs) puis au nettoyage via NLTK (suppression des stopwords, lemmatisation, protection des termes techniques).
Le texte nettoyé est passé simultanément aux deux modèles chargés en mémoire (Classique SVM/LR et Advanced SBERT) pour prédiction.
L'API retourne un objet JSON structuré contenant les deux prédictions, les scores de confiance, un extrait du texte, les mots-clés, et le statut d'accord des modèles.
Enfin, le frontend React analyse ce JSON et affiche ces résultats dynamiquement, incluant la comparaison, les détails de confiance et met à jour l'historique de la session.
