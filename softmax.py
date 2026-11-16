"""NumPy implementation of 3-class softmax regression, trained with batch GD.

The model (not text vectorization or evaluation) is implemented from scratch.
Run: python3 softmax.py
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from common import ROOT, CLASSES, load_data, split_data, text_features, metric_row


class SoftmaxScratch:
    def __init__(self, learning_rate=0.5, regularization=0.01, epochs=500):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs

    def fit(self, X, y):
        ids = np.searchsorted(CLASSES, np.asarray(y))
        n, d = X.shape
        k = len(CLASSES)
        self.W = np.zeros((d, k), dtype=float)
        self.b = np.zeros(k, dtype=float)
        onehot = np.eye(k)[ids]
        self.loss_history = []
        for epoch in range(self.epochs):
            scores = np.asarray(X @ self.W + self.b)
            scores -= scores.max(axis=1, keepdims=True)
            e = np.exp(scores)
            prob = e / e.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(np.maximum(prob[np.arange(n), ids], 1e-15)))
            loss += 0.5 * self.regularization * np.sum(self.W ** 2)
            self.loss_history.append(float(loss))
            delta = prob - onehot
            grad_W = np.asarray(X.T @ delta) / n + self.regularization * self.W
            grad_b = delta.mean(axis=0)
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
        return self

    def predict(self, X):
        scores = np.asarray(X @ self.W + self.b)
        return CLASSES[np.argmax(scores, axis=1)]


def make_features(train, val, ngrams=1):
    vec = CountVectorizer(ngram_range=(1, ngrams), min_df=2)
    X_train = vec.fit_transform(text_features(train))
    X_val = vec.transform(text_features(val))
    return X_train, X_val, vec


def run_one(train, val, regularization=0.01, ngrams=1, epochs=500):
    X_train, X_val, vec = make_features(train, val, ngrams)
    model = SoftmaxScratch(regularization=regularization, epochs=epochs).fit(
        X_train, train.Stance.to_numpy())
    pred_tr = model.predict(X_train)
    pred_val = model.predict(X_val)
    tr = metric_row('Scratch softmax', train.Stance, pred_tr)
    va = metric_row('Scratch softmax', val.Stance, pred_val)
    return model, tr, va, len(vec.vocabulary_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()
    df = load_data()
    train, val = split_data(df)
    print(f'Rows: {len(df)} | train: {len(train)} | validation: {len(val)}')
    Xtr, Xva, vec = make_features(train, val)
    reference = [
        ('Majority', DummyClassifier(strategy='most_frequent')),
        ('sklearn logistic', LogisticRegression(max_iter=1000)),
    ]
    rows = []
    for name, clf in reference:
        clf.fit(Xtr, train.Stance)
        for subset, X, y in [('train', Xtr, train.Stance), ('validation', Xva, val.Stance)]:
            rows.append({'model': name, 'split': subset, **metric_row(name, y, clf.predict(X)),
                         'lambda': np.nan, 'vocabulary': Xtr.shape[1]})
    sweep = [0.0, 0.0001, 0.001, 0.01, 0.1]
    for lam in sweep:
        model = SoftmaxScratch(regularization=lam, epochs=args.epochs).fit(Xtr, train.Stance)
        for subset, X, y in [('train', Xtr, train.Stance), ('validation', Xva, val.Stance)]:
            rows.append({'model': 'Scratch softmax', 'split': subset,
                         **metric_row('Scratch softmax', y, model.predict(X)),
                         'lambda': lam, 'vocabulary': len(vec.vocabulary_)})
        print(f'lambda={lam:g}: final loss={model.loss_history[-1]:.4f}; '
              f'validation Favg={rows[-1]["f_avg"]:.4f}')
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'regularization_results.csv', index=False)
    plot_data = results[results.model == 'Scratch softmax']
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(sweep))
    for subset in ('train', 'validation'):
        vals = [plot_data[(plot_data['split'] == subset) & (plot_data['lambda'] == lam)]
                .iloc[0]['f_avg'] for lam in sweep]
        ax.plot(positions, vals, marker='o', label=subset.title())
    ax.set_xticks(positions, ['0', '0.0001', '0.001', '0.01', '0.1'])
    ax.set_xlabel('L2 regularization strength (lambda)')
    ax.set_ylabel('Favg (FAVOR/AGAINST)')
    ax.set_title('L2 regularization sweep (unigrams)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'regularization_sweep.png', dpi=200)
    plt.close(fig)
    print('\n' + results[['model', 'split', 'lambda', 'f_avg', 'accuracy']].to_string(index=False))

    # M2: bigrams, held constant at lambda=0.01 for a controlled feature comparison.
    comparison = []
    for ngrams in (1, 2):
        model, tr, va, vocab_size = run_one(train, val, regularization=0.01,
                                            ngrams=ngrams, epochs=args.epochs)
        comparison.append({'features': 'unigrams' if ngrams == 1 else 'unigrams+bigrams',
                           'vocab_size': vocab_size, 'train_favg': tr['f_avg'],
                           'val_favg': va['f_avg'], 'val_accuracy': va['accuracy']})
    comp = pd.DataFrame(comparison)
    comp.to_csv(ROOT / 'ngram_comparison.csv', index=False)
    print('\nN-gram comparison (lambda=0.01):\n', comp.to_string(index=False))
    print('\nSaved regularization_results.csv, regularization_sweep.png, ngram_comparison.csv')


if __name__ == '__main__':
    main()
