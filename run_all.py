"""
Full pipeline runner.
Runs all steps in order. Stops immediately if any step fails.

Usage:
    python run_all.py            # runs everything (extract steps included)
    python run_all.py --skip-extract   # skips feature extraction
"""
import subprocess
import sys

STEPS = [
    ("Feature extraction — RDKit descriptors",      "extract_features.py"),
    ("Feature extraction — ChemBERTa embeddings",   "extract_transformer_features.py"),
    ("Baseline model comparison",                    "train_models.py"),
    ("Hyperparameter tuning",                        "tuning_models.py"),
    ("Global SHAP explanations",                     "explain_shap.py"),
    ("Local SHAP explanations",                      "explain_local.py"),
    ("Model evaluation (curves + report)",           "evaluate_models.py"),
    ("Visualisation",                                "visualize_results.py"),
]

EXTRACT_SCRIPTS = {"extract_features.py", "extract_transformer_features.py"}

skip_extract = "--skip-extract" in sys.argv

total = len([s for s in STEPS if not (skip_extract and s[1] in EXTRACT_SCRIPTS)])
run = 0

for name, script in STEPS:
    if skip_extract and script in EXTRACT_SCRIPTS:
        continue

    run += 1
    print(f"[{run}/{total}] {name} ...")

    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"[ERROR] '{name}' failed (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)

print("Pipeline completed successfully.")
