# Stance project: progress report scripts

1. Copy `common.py`, `softmax.py`, `analysis.py` and `half_data.py` into the **same directory** as your original `baseline (1).py` and `training_data.tsv` (if downloaded).
2. In Terminal open that directory, e.g. `cd ~/Desktop/Stance-Classifier-main`.
3. Install dependencies if needed: `python3 -m pip install numpy pandas scipy scikit-learn matplotlib requests`.
4. Run in order:

   ```bash
   python3 softmax.py
   python3 analysis.py
   python3 half_data.py
   ```

## Expected artifacts

- `regularization_results.csv`: NumPy softmax L2 sweep and sklearn reference metrics.
- `regularization_sweep.png`: model training vs validation Favg as L2 varies.
- `ngram_comparison.csv`: unigram vs unigram+bigram scratch softmax (L2=0.01).
- `per_target_results.csv`: topic-level Favg and accuracy for sklearn/scratch.
- `confusion_matrices.png`: actual vs predicted classes on validation set.
- `half_data_results.csv`: five half-size data trials and one full-data run.
- `half_data_comparison.png`: comparison plot.

## Important scientific notes

- All results are computed using the **same stratified validation split** as your original baseline. Official test data is not downloaded or used here.
- `SoftmaxScratch` is implemented using NumPy gradient descent and L2. `CountVectorizer`, split functions, and `LogisticRegression` references are scikit-learn utilities.
- The five half-size training subsets are independently sampled from the development training split, stratified by topic and stance. They may overlap, which matches the proposed design.
- The 50%/100% comparison uses fixed hyperparameters selected beforehand; do not tune hyperparameters based on the five validation comparisons.
- Be careful not to claim the half-data experiment proves the final hypothesis on the official test set; it is preliminary validation evidence.
- If your teammate has already built `common.py` or `softmax.py`, compare files before replacing anything. Don't overwrite teammate changes.
- Commit scripts and generated figures/CSV files, but don't commit downloaded tweet data until redistribution permissions are confirmed.
