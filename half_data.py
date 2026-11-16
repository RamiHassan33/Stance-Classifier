"""Repeat five 50%-vs-100% training experiments on one fixed validation set.

These are exploratory VALIDATION results, not official test-set findings.
Hyperparameters are fixed in advance (no selection on this comparison).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from common import ROOT, load_data, split_data
from softmax import run_one

SEEDS = [11, 22, 33, 44, 55]
REGULARIZATION = 0.01
EPOCHS = 500


def main():
    df = load_data()
    train, val = split_data(df)
    # Fixed validation. Each sample is half of the development-training pool.
    # Feature vocabulary is fitted separately for each pool (no leakage).
    _, full_tr, full_val, full_vocab = run_one(
        train, val, regularization=REGULARIZATION, ngrams=1, epochs=EPOCHS)
    rows = [{'fraction': 1.0, 'seed': 'full', 'n_train': len(train),
             'vocab_size': full_vocab, 'train_favg': full_tr['f_avg'],
             'val_favg': full_val['f_avg'], 'val_accuracy': full_val['accuracy']}]
    for seed in SEEDS:
        half, _ = train_test_split(
            train, train_size=0.5, random_state=seed,
            stratify=train['Target'] + '|' + train['Stance'])
        _, tr, va, vocab = run_one(
            half, val, regularization=REGULARIZATION, ngrams=1, epochs=EPOCHS)
        rows.append({'fraction': 0.5, 'seed': seed, 'n_train': len(half),
                     'vocab_size': vocab, 'train_favg': tr['f_avg'],
                     'val_favg': va['f_avg'], 'val_accuracy': va['accuracy']})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'half_data_results.csv', index=False)
    half_scores = out.loc[out.fraction == 0.5, 'val_favg'].to_numpy()
    delta = float(full_val['f_avg'] - half_scores.mean())
    print(out.to_string(index=False))
    print(f'\n100% Favg: {full_val["f_avg"]:.4f}')
    print(f'50% mean Favg: {half_scores.mean():.4f}')
    print(f'50% standard deviation (sample): {half_scores.std(ddof=1):.4f}')
    print(f'50% range: {half_scores.min():.4f} to {half_scores.max():.4f}')
    print(f'100% minus 50%-mean: {delta:+.4f}')
    print('Exploratory hypothesis threshold (+0.02):',
          'met on validation' if delta >= 0.02 else 'not met on validation')
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(np.repeat(0, len(half_scores)), half_scores, label='Five 50% subsets')
    ax.scatter([1], [full_val['f_avg']], marker='D', label='100% training pool')
    ax.set_xticks([0, 1], ['50% (5 trials)', '100%'])
    ax.set_ylabel('Validation Favg')
    ax.set_title('Effect of training-set size (exploratory validation)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'half_data_comparison.png', dpi=200)
    plt.close(fig)
    print('\nSaved half_data_results.csv and half_data_comparison.png')


if __name__ == '__main__':
    main()
