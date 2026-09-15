# Πρόβλεψη ιδιοτήτων φαρμάκων με βάση τη χημική τους δομή και παραγωγή εξηγήσεων

Πτυχιακή εργασία για την πρόβλεψη της τοξικότητας φαρμακευτικών ενώσεων από τη
μοριακή τους δομή (SMILES). Το ερευνητικό ερώτημα είναι κατά πόσο χρειάζεται ένας Transformer, ή αν οι κλασικοί RDKit derscriptors τα καταφέρνουν εξίσου καλά στην πρόβλεψη τοξικότητας μορίων. Αυτό επιτυχγάνεται μέσω της σύγκρισης των δύο τρόπων αναπαράστασης των μορίων: των RDKit descriptors και των ChemBERTa embeddings. Ακόμη, στο πείραμα, παρουσιάζεται και η εξήγηση των αντίστοιχων επιδόσεων με χρήση SHAP.

## Δεδομένα

Τρία datasets τοξικότητας από το TDC (Therapeutics Data Commons), με scaffold
split σε train/valid/test:

- `carcinogens` : καρκινογένεση
- `clintox` : κλινική τοξικότητα
- `skin_reaction` : δερματική αντίδραση

Για κάθε dataset εξάγονται δύο feature sets, τα οποία αποθηκεύονται στον
φάκελο `data/`:

- RDKit descriptors (`*_rdkit.csv`)
- ChemBERTa embeddings (`*_chemberta.csv`)

## Μοντέλα

Κάθε συνδυασμός dataset και feature set δοκιμάζεται σε τρία μοντέλα:

- Logistic Regression
- Random Forest
- XGBoost

Ακολουθεί hyperparameter tuning και αξιολόγηση με ROC-AUC και PR-AUC.

## Δομή του κώδικα

| Αρχείο | Τι κάνει |
|---|---|
| `run_all.py` | Τρέχει όλο το pipeline με τη σωστή σειρά |
| `utils.py` | Βοηθητικές συναρτήσεις (φόρτωση δεδομένων, κατασκευή pipeline, κ.α.) |
| `extract_features.py` | Εξαγωγή RDKit descriptors |
| `extract_transformer_features.py` | Εξαγωγή ChemBERTa embeddings |
| `train_models.py` | Baseline εκπαίδευση και σύγκριση μοντέλων |
| `tuning_models.py` | Hyperparameter tuning |
| `explain_shap.py` | Global explainability |
| `explain_local.py` | Local explainability (SHAP ανά μόριο) |
| `evaluate_models.py` | Αξιολόγηση: ROC/PR curves και classification report |
| `visualize_results.py` | Συγκεντρωτικά γραφήματα των αποτελεσμάτων |

## Εκτέλεση

Εγκατάσταση των εξαρτήσεων:

```bash
pip install -r requirements.txt
```

Εκτέλεση όλου του pipeline:

```bash
python run_all.py
```

Αν τα χαρακτηριστικά έχουν ήδη εξαχθεί, μπορούν να παραλειφθούν τα βήματα:

```bash
python run_all.py --skip-extract
```

## Αποτελέσματα

Τα αποτελέσματα αποθηκεύονται στον φάκελο `results/`:

- `model_results.csv` — τελικές επιδόσεις των μοντέλων μετά το tuning
- `baseline_comparison_results.csv` — επιδόσεις των baseline μοντέλων
- `evaluation/` — ROC/PR curves και confusion matrices
- `shap/` — global SHAP plots και top features ανά dataset
- `shap/local/` — τοπικές εξηγήσεις για μεμονωμένα μόρια
- `figures/` — συγκεντρωτικά γραφήματα σύγκρισης
