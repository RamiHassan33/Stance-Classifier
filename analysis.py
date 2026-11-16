"""Per-topic Favg and confusion matrix for baseline and scratch model."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from common import ROOT, CLASSES, load_data, split_data, metric_row
from softmax import SoftmaxScratch, make_features


def main():
    df = load_data()
    train, val = split_data(df)
    Xtr, Xva, _ = make_features(train, val, ngrams=1)
    models = [
        ('sklearn unigram', LogisticRegression(max_iter=1000).fit(Xtr, train.Stance)),
        ('scratch unigram (lambda=0.01)',
         SoftmaxScratch(regularization=0.01).fit(Xtr, train.Stance)),
    ]
    topic_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, model) in zip(axes, models):
        predicted = model.predict(Xva)
        for target in sorted(val.Target.unique()):
            mask = (val.Target.to_numpy() == target)
            row = metric_row(name, val.Stance.to_numpy()[mask], predicted[mask])
            topic_rows.append({'model': name, 'target': target, 'n': int(mask.sum()),
                               'f_avg': row['f_avg'], 'accuracy': row['accuracy']})
        cm = confusion_matrix(val.Stance, predicted, labels=CLASSES)
        img = ax.imshow(cm, cmap='Blues')
        ax.set_title(name)
        ax.set_xticks(range(3), CLASSES, rotation=25)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel('Predicted stance')
        ax.set_ylabel('True stance')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        print(f'{name}: validation Favg={metric_row(name, val.Stance, predicted)["f_avg"]:.4f}')
    fig.tight_layout()
    fig.savefig(ROOT / 'confusion_matrices.png', dpi=200)
    plt.close(fig)
    out = pd.DataFrame(topic_rows)
    out.to_csv(ROOT / 'per_target_results.csv', index=False)
    print('\nPer-target metrics:\n', out.to_string(index=False))
    print('\nSaved confusion_matrices.png and per_target_results.csv')


if __name__ == '__main__':
    main()
