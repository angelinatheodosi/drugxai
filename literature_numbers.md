# Νούμερα από τη βιβλιογραφία

Όλα τα νούμερα είναι **ROC-AUC στο test set**, εκτός αν αναφέρεται κάτι άλλο.

## Τα δικά μου αποτελέσματα

- **Split:** scaffold του TDC, `get_split(method="scaffold")` με τα defaults seed=42 και frac=[0.7, 0.1, 0.2].
- **Μεγέθη:** ClinTox 1,478 μόρια (μόνο η ετικέτα CT_TOX), Carcinogens 280, Skin Reaction 404.
- **Features:** RDKit descriptors και frozen ChemBERTa embeddings (`seyonec/ChemBERTa-zinc-base-v1`).
- **Πηγή:** `results/model_results.csv`, στήλη `test_ROC-AUC`.
- **Επιλογή καλύτερου μοντέλου:** με το υψηλότερο **validation PR-AUC** (`utils.load_best_models`, `results/figures/best_models_summary.csv`). Το test set **δεν** χρησιμοποιείται για την επιλογή.

### Καλύτερο μοντέλο ανά dataset και feature set

| Dataset | RDKit | ChemBERTa |
|---|---|---|
| ClinTox | **0.811** (LR) | 0.723 (XGB) |
| Carcinogens | **0.809** (LR) | 0.797 (LR) |
| Skin Reaction | 0.664 (LR) | **0.667** (RF) |

### Όλα τα μοντέλα (LR / RF / XGB)

Με * σημειώνεται το μοντέλο που επιλέγεται με validation PR-AUC.

| Dataset | RDKit: LR / RF / XGB | ChemBERTa: LR / RF / XGB |
|---|---|---|
| ClinTox | 0.811* / 0.829 / 0.770 | 0.710 / 0.642 / 0.723* |
| Carcinogens | 0.809* / 0.818 / 0.807 | 0.797* / 0.794 / 0.664 |
| Skin Reaction | 0.664* / 0.673 / 0.658 | 0.498 / 0.667* / 0.615 |

---

## Σημείωση: οι δύο εκδοχές του ClinTox

- Το ClinTox του **MoleculeNet** έχει 2 tasks: **FDA_APPROVED** (αν εγκρίθηκε από τον FDA) και **CT_TOX** (αν απέτυχε σε κλινικές δοκιμές λόγω τοξικότητας).
- Το ClinTox του **TDC**, που χρησιμοποιώ, έχει **μόνο το CT_TOX**.
- Όσα άρθρα αναφέρουν τον μέσο όρο των 2 tasks **δεν** μετρούν ακριβώς το ίδιο πρόβλημα με μένα.

| Άρθρο | Τι μετρά στο ClinTox | Ίδιο με το δικό μου; |
|---|---|---|
| Chu & Cho | ClinTox του TDC (CT_TOX) | ✅ |
| ChemBERTa paper | μόνο CT_TOX | ✅ |
| Gray & Wu | μόνο «toxic / non-toxic» | ✅ |
| Born et al. | μέσος όρος 2 tasks | ⚠️ |
| MoleculeNet | μέσος όρος 2 tasks | ⚠️ |
| Sharma et al. | μέσος όρος tasks (έμμεσα) | ⚠️ |

## Σημείωση: seeds και split

- Το scaffold split του TDC **εξαρτάται από seed**. Ο κώδικας (`tdc/utils/split.py`, `create_scaffold_split`) ομαδοποιεί τα μόρια ανά scaffold, ανακατεύει τις ομάδες με βάση το seed και μετά τις μοιράζει. Η προεπιλογή είναι seed=42.
- Το «± std» στα άρθρα με σταθερό split (π.χ. FATE-Tox) μετρά μόνο την **τυχαιότητα της εκπαίδευσης** (αρχικοποίηση βαρών, dropout, bootstrap του RF). Δεν μετρά το πόσο θα άλλαζε το αποτέλεσμα με άλλα δεδομένα. Ένδειξη γι' αυτό: το SVM, που είναι σχεδόν ντετερμινιστικό, έχει std ≈ 0.
- Εγώ έχω 1 split και 1 run. Με test sets των ~56 μορίων (Carcinogens) και ~82 μορίων (Skin), διαφορές μικρότερες από ~0.03 δεν είναι σημαντικές.

---

## 1. Chu & Cho (2024), IEEE BIBM

- **Split:** «predetermined split» του TDC. Δεν δηλώνεται η μέθοδος, πιθανότατα η προεπιλεγμένη random.
- **Features για τα κλασικά μοντέλα:** ECFP fingerprints.
- **Πηγή:** Table II.

| Dataset | Καλύτερο | LR | RF | XGB |
|---|---|---|---|---|
| ClinTox | MolFormer (Transformer) **0.946** | 0.842 | 0.763 | 0.886 |
| Skin Reaction | MolFormer = SVM **0.789** | 0.727 | 0.755 | 0.758 |
| Carcinogens | MolCLR-gin (γράφος) **0.883** | 0.841 | 0.800 | 0.788 |

**Προσοχή:** στον Table I γράφουν Carcinogens = 2,870 και ClinTox = 7,265 μόρια. Είναι λάθος. Το TDC έχει 280 και 1,478 μόρια αντίστοιχα.

## 2. FATE-Tox: Ha, Bang & Kim (2025), Journal of Cheminformatics

- **Split:** τα έτοιμα scaffold splits του TDC. Γράφουν: «we follow the scaffold-splitted subsets given in the datasets».
- 3 seeds, με το std σε παρένθεση.
- **Πηγή:** Table 2.
- Δεν δηλώνουν ποια features χρησιμοποιούν τα RF/SVM ούτε αν έγινε tuning. Δεν έχει LR ούτε XGB.

| Μοντέλο | Skin Reaction | Carcinogens |
|---|---|---|
| FATE-Tox MTL, καλύτερο | **0.741** (0.008) | **0.848** (0.003) |
| FATE-Tox STL | 0.733 (0.006) | 0.842 (0.021) |
| MAT | 0.659 (0.011) | 0.830 (0.035) |
| Uni-Mol | 0.695 (0.048) | 0.822 (0.035) |
| GraphMVP | 0.613 (0.035) | 0.795 (0.047) |
| SVM | 0.731 (0.000) | 0.783 (0.001) |
| RF | 0.677 (0.021) | 0.751 (0.032) |
| MLP | 0.499 (0.138) | 0.783 (0.002) |
| MolCLR | 0.451 (0.062) | 0.740 (0.038) |
| MolFormer (Transformer σε SMILES) | 0.643 (0.016) | 0.730 (0.002) |

## 3. TxGemma (Google, 2025, arXiv 2504.06196) και Tx-LLM (Google, 2024, arXiv 2406.06316)

- **Split:** scaffold του TDC. Skin: 282 / 40 / 82 μόρια σε train / valid / test. Carcinogens: 196 / 28 / 56.
- Μεγάλα γλωσσικά μοντέλα (LLMs) που εκπαιδεύτηκαν σε όλο το TDC.
- Για το Carcinogens δίνουν **accuracy** αντί για AUROC, οπότε δεν συγκρίνεται.
- Το «specialist SOTA» 0.840 για το Skin προέρχεται από το αρχικό άρθρο του dataset (Alves 2015), με άλλο setup. **Δεν** συγκρίνεται.
- **Πηγή:** TxGemma Tables S.13, S.15, S.17.

| Skin Reaction | AUROC |
|---|---|
| TxGemma-9B-Predict (commercial), καλύτερο | **0.733** |
| TxGemma-27B-Predict | 0.708 |
| Tx-LLM M | 0.615 |
| Tx-LLM S | 0.564 |

## 4. Gray & Wu (2025), Chem. Res. Toxicol.

Είναι το πιο κοντινό στο δικό μου setup.

- **Split:** scaffold. Υποσύνολο ClinTox με 655 μόρια, από τα οποία 64 στο test.
- **Setup:** RDKit/Mordred descriptors και frozen LM embeddings, με LR ή RF από πάνω. Ίδιο ChemBERTa checkpoint με το δικό μου.
- **Πηγή:** κυρίως άρθρο για το LR, Figure S1 του supplementary για το RF.

| ClinTox | LR | RF (≈, διαβασμένο από γράφημα) |
|---|---|---|
| RDKit | 0.721 | ≈ 0.77 |
| Mordred | 0.604 | ≈ 0.70 |
| Καλύτερο LM από SMILES (BERT) | 0.900 | ≈ 0.92 |
| Καλύτερο LM γενικά (GPT-3 σε περιγραφές) | 0.996 | ≈ 0.999 |

- Το ChemBERTa για το ClinTox **δεν δίνεται** πουθενά, ούτε στο supplementary.
- DILIst (Table S2, LR, 5 seeds): RDKit 0.618 ± 0.037 έναντι ChemBERTa από SMILES 0.562 ± 0.055.
- Tox21 (Table S1): το ChemBERTa είναι το LM με τη μεγαλύτερη απόκλιση από τα descriptors (MAZ 3.94 έναντι Mordred).

## 5. ChemBERTa (Chithrananda et al., 2020)

- **Split:** scaffold 80/10/10 (DeepChem).
- **Task:** μόνο το CT_TOX του ClinTox.
- **Setup:** ChemBERTa με fine-tuning. Τα RF και SVM τρέχουν σε 2048-bit Morgan fingerprints (Chemprop).
- **Πηγή:** Table 1.

| ClinTox (CT_TOX) | ROC | PRC |
|---|---|---|
| D-MPNN (γράφος), καλύτερο | **0.906** | 0.993 |
| SVM | 0.833 | 0.986 |
| ChemBERTa 10M | 0.733 | 0.975 |
| RF | 0.693 | 0.968 |

## 6. Born et al. (2023), Digital Discovery (ToxSmi)

- **Πηγή:** Table 2 του άρθρου για το random split, Table A3 του supplementary για το scaffold split.
- Μέσος όρος των 2 tasks του ClinTox (⚠️ βλ. σημείωση παραπάνω).
- Δεν έχει LR/RF/XGB.

| ClinTox | Random split (10 επαναλήψεις) | Scaffold split |
|---|---|---|
| AttentiveFP (γράφος) | — | **0.933 ± 0.020** |
| GROVER (graph transformer) | — | 0.925 ± 0.013 |
| D-MPNN | **0.894** | — |
| CLM (δικό τους, SMILES) | 0.878 | 0.896 ± 0.006 |

Με scaffold split, ο CLM έχει 10 επαναλήψεις. Τα υπόλοιπα μοντέλα έχουν 3 επαναλήψεις και οι τιμές τους προέρχονται από το GROVER paper (Rong et al.).

## 7. MoleculeNet (Wu et al., 2018)

- **Split:** random.
- Μέσος όρος των 2 tasks του ClinTox (⚠️).
- **Πηγή:** Table 3.

| ClinTox | ROC-AUC |
|---|---|
| Weave (γράφος), καλύτερο | **0.832** |
| Bypass (νευρωνικό σε ECFP), καλύτερο «κλασικό» | 0.827 |

Τα LR/RF/XGB υπάρχουν μόνο σε γράφημα (Fig. 9), χωρίς αριθμούς.

## 8. Sharma et al. (2023), Scientific Reports

- **Split:** 80/10/10, χωρίς να δηλώνεται η μέθοδος.
- Μέσος όρος tasks (⚠️). Δεν έχει LR/RF/XGB.

| ClinTox | ROC-AUC |
|---|---|
| Multi-task DNN με SMILES embeddings, καλύτερο | **0.994 ± 0.005** |
| Single-task DNN με SMILES embeddings | 0.987 ± 0.019 |
| Weave (baseline από MoleculeNet) | 0.832 ± 0.037 |

## TDC (Huang et al., 2021)

Δεν δίνει αποτελέσματα για τα 3 datasets. Ορίζει όμως το προτεινόμενο πρωτόκολλο: **scaffold split + AUROC**. Αναφέρει μεγέθη: Skin Reaction 404, Carcinogens 278, ClinTox 1,484.

---

## Άμεσες συγκρίσεις (ίδιο split scaffold, ίδιο task)

| Dataset | Πηγή | Κλασικό μοντέλο | Transformer σε SMILES | Καλύτερο του άρθρου |
|---|---|---|---|---|
| **ClinTox** | **Εγώ** | RDKit + LR **0.811** | ChemBERTa (frozen) + XGB 0.723 | — |
| | Gray & Wu | RDKit: LR 0.721 / RF ≈0.77 | — | BERT ≈0.92 |
| | ChemBERTa paper | RF (Morgan) 0.693 | ChemBERTa (fine-tuned) 0.733 | D-MPNN 0.906 |
| **Carcinogens** | **Εγώ** | RDKit + LR **0.809** | ChemBERTa (frozen) + LR 0.797 | — |
| | FATE-Tox | RF 0.751, SVM 0.783 | MolFormer 0.730 | FATE-Tox 0.848 |
| **Skin Reaction** | **Εγώ** | RDKit + LR 0.664 | ChemBERTa (frozen) + RF **0.667** | — |
| | FATE-Tox | RF 0.677, SVM 0.731 | MolFormer 0.643 | FATE-Tox 0.741 |
| | TxGemma | — | — | TxGemma-9B 0.733 |

## Συμπεράσματα

1. **ClinTox:** το RDKit μου (0.811) είναι πάνω από το RDKit του Gray & Wu (0.721 / ≈0.77). Το frozen ChemBERTa μου (0.723) είναι στο ίδιο επίπεδο με το fine-tuned ChemBERTa του αρχικού paper (0.733).
2. **Carcinogens:** το RDKit μου (0.809) είναι πάνω από τα RF (0.751) και MolFormer (0.730) του FATE-Tox.
3. **Skin Reaction:** RDKit και ChemBERTa πρακτικά ισοπαλία (0.664 / 0.667), στο ίδιο επίπεδο με το RF του FATE-Tox (0.677). Το dataset είναι δύσκολο για όλα τα μοντέλα, που κινούνται στο 0.64–0.74.
4. **Ο Transformer σε SMILES δεν ξεπερνά τα κλασικά features με scaffold split:** το ChemBERTa μου, ο MolFormer του FATE-Tox και το ChemBERTa του αρχικού paper.
5. **Πάνω από 0.9 φτάνουν μόνο** τα μοντέλα γράφων με pretraining (AttentiveFP, GROVER, D-MPNN), τα μεγάλα μοντέλα με fine-tuning (MolFormer στο Chu & Cho), η εξωτερική πληροφορία (κείμενο, multi-task) ή το random split.
6. Το μόνο σαφές πλεονέκτημα Transformer στη βιβλιογραφία είναι στο ClinTox (Chu & Cho, 0.946). Εκεί όμως το split δεν δηλώνεται και ο MolFormer έχει fine-tuning.

## Περιορισμοί του δικού μου πειράματος

- 1 split και 1 run, χωρίς std. Αν δοκιμάσω πολλά seeds στο `get_split`, θα έχω μέσο όρο ± std.
- Frozen embeddings και όχι fine-tuning του ChemBERTa.
- Σχεδόν κανένα άρθρο δεν δίνει PR-AUC (εξαίρεση το ChemBERTa paper για το ClinTox), οπότε για αυτή τη μετρική δεν υπάρχει σημείο σύγκρισης.
