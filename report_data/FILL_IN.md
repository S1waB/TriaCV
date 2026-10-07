# Tableau Récapitulatif (Internship Report)

| Élément à compléter | Valeur mesurée | Fichier source |
| --- | --- | --- |
| **Statistiques du dataset** (nb CV, nb cat., maj/min, longueur, >512, doublons) | 1250 CVs, 25 catégories (parfaitement équilibrées : 50 CVs chacune, ratio 1.0). Longueur médiane : 258 mots. 0.0% > 512 tokens. Doublons : 0 exacts, 0 quasi. | `eda_stats.json` |
| **Split & Validation** (proportion, graine, plis CV) | Split 80/20 (stratifié), Graine (random seed) : 42, 5 plis (folds) de validation croisée. | `project_facts.json`, code `run_baseline.py` |
| **Techniques & Matériel** (lib prétraitement, SBERT, DistilBERT, hardware) | NLTK (English). SBERT : `all-MiniLM-L6-v2`. DistilBERT : `distilbert-base-uncased`. Matériel : Intel Core i7-13700HX (Exécuté sur CPU uniquement). | `project_facts.json`, `transformers_hyperparams.json` |
| **Hyperparamètres Baseline** (TF-IDF, LR, SVM, NB, RF) | TF-IDF : n-grams (1,2), max_features=30000. LR : C=0.1, lbfgs. SVM : C=0.1. NB : alpha=0.01. RF : max_depth=None, min_samples_split=2. | `baseline_hyperparams.json` |
| **Hyperparamètres DistilBERT** (lr, batch, ep, max_len, wd, warmup) | lr=2e-5, batch_size=4, epochs=2 (réduit pour CPU), max_length=512, weight_decay=0.01. (Warmup : NOT MEASURED/Retiré pour compatibilité). | `transformers_hyperparams.json` |
| **Métriques (7 colonnes, 6 modèles)** | **LR** : Acc 1.0, Pr 1.0, Rc 1.0, F1m 1.0, F1w 1.0, Train 7.02s, Inf 0.01ms<br>**SVM** : Acc 1.0, Pr 1.0, Rc 1.0, F1m 1.0, F1w 1.0, Train 1.93s, Inf 0.01ms<br>**NB** : Acc 1.0, Pr 1.0, Rc 1.0, F1m 1.0, F1w 1.0, Train 0.23s, Inf 0.01ms<br>**RF** : Acc 1.0, Pr 1.0, Rc 1.0, F1m 1.0, F1w 1.0, Train 3.53s, Inf 0.17ms<br>**SBERT+LR** : Acc 1.0, Pr 1.0, Rc 1.0, F1m 1.0, F1w 1.0, Train 20.94s, Inf 12.78ms<br>**DistilBERT** : Acc 1.0, Pr 1.0, Rc 1.0, F1m 1.0, F1w 1.0, Train 1420.89s, Inf 64.85ms | `comparison_table.csv` |
| **Modèles (Classique, Transformer, Déployé)** | Meilleur Classique : Logistic Regression (à égalité). Meilleur Transformer : Sentence-BERT. Déployé : Logistic Regression (TF-IDF) justifié par sa taille (<1MB) et son inférence instantanée (<1ms) pour une performance identique (1.0). (Note: L'API affiche "Linear SVM"). | `comparison_summary_fr.md` |
| **Confusions (Baseline & Transformer)** | Aucune confusion pour aucun des deux modèles (Précision parfaite 100% sur le test set). | `misclassified_examples.md`, `transformers_summary_fr.md` |
| **Termes discriminants (3 catégories)** | *Electrical Engineering* : electrical, engineering, system. *DotNet Developer* : sql, dotnet, developer. *PMO* : project, management, tracking. | `top_terms_per_class.csv` / `eda_stats.json` |
| **Résultats T1 à T8** | T2(texte), T3(vide), T4(png), T5(lourd), T7(comparaison JSON), T8(history) : PASS. T1(PDF valide) & T6(PDF scanné) : FAIL (Fichiers de test non disponibles). | `tests_results.csv` |
| **Liste des PNG générés avec légende** | `fig_distribution_categories.png` (Répartition des CV), `fig_longueur_cv.png` (Longueur en mots), `fig_top_termes.png` (Mots fréquents), `fig_confusion_baseline.png` (Matrice Classique), `fig_confusion_transformer.png` (Matrice Transformer), `fig_courbes_distilbert.png` (Perte Train/Val), `fig_comparaison_f1.png` (F1-Macro Comparés), `fig_roc.png` (Courbes ROC), `fig_diff_f1_par_classe.png` (Gain par classe). (Captures UI manquantes). | Dossier `report_data/` |
| **Routes API** | `POST /api/predict`, `GET /api/history` | `project_facts.json` |
| **Livrables (Statut & Chemin)** | Notebooks (MISSING), Données (FOUND: `data/raw/`, `data/processed/`), Modèles (FOUND: `models_artifacts/`), API Backend (FOUND: `api/`), Web Frontend (FOUND: `frontend/`), Slides (MISSING). | `project_facts.json` |

---

## Éléments manquants ou échoués (NOT MEASURED / FAIL)

Afin d'être parfaitement transparent, voici les éléments qui n'ont pas pu être mesurés ou qui ont échoué lors de cette session d'évaluation :

1. **Test T1 (PDF Valide) & Test T6 (PDF Scanné) :** **FAIL / NOT MEASURED**. Aucun fichier PDF de test n'était disponible dans l'environnement pour effectuer la requête API réelle, bien que le code de l'API (`extractor.py`) implémente bien cette fonctionnalité avec `PyMuPDF`.
2. **Captures d'écran UI (React) :** **NOT MEASURED**. L'agent de navigation web a échoué à cause de l'impossibilité d'installer le driver du navigateur automatisé Playwright sur cette machine (Erreur HTTP 404 du CDN Azure). Les 6 images UI (`capture_accueil.png`, etc.) ne sont donc pas générées.
3. **Paramètre "Warmup" de DistilBERT :** **NOT MEASURED**. Ce paramètre a dû être retiré des hyperparamètres car la version actuelle de la librairie `transformers` installée sur l'environnement levait une exception `TypeError` à l'initialisation.
4. **Livrables (Slides et Notebooks) :** **NOT MEASURED / MISSING**. Le dossier de l'application ne contient ni les notebooks d'expérimentation ni les slides de présentation finale.
5. **Confusions et Différence par classe :** Les "3 principales confusions" et les gains de F1 ne sont pas présents, tout simplement car l'ensemble de test a été classifié à 100% sans erreur de prédiction par tous les modèles.
