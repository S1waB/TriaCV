# Résumé Transformers

> **Note matérielle :** Exécuté sur CPU uniquement. Les paramètres de DistilBERT ont été réduits (2 époques, batch=4) pour terminer l'entraînement dans un temps raisonnable (1420 secondes).

Le meilleur modèle basé sur les Transformers est **Sentence-BERT + LR** avec un F1-macro parfait de 1.0000, égalant DistilBERT tout en étant beaucoup plus léger.
Sentence-BERT obtient un F1-macro de 1.0000 sur texte prétraité, avec une ablation révélant que le texte brut donne également un score de 1.0000.
DistilBERT (fine-tuné) obtient un F1-macro de 1.0000 avec un écart Train/Val Loss de -0.2282, montrant une excellente généralisation malgré peu d'époques.
L'inférence de Sentence-BERT prend environ 12.8 ms par CV, contre 64.9 ms pour DistilBERT, offrant un bien meilleur compromis performance/temps.
Le modèle a atteint une précision parfaite sur cet ensemble de test (comme pour les modèles classiques), ne produisant aucune erreur de classification.
