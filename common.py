"""Shared SemEval Task A loading, reproducible split, and stance metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'training_data.tsv'
DATA_URL = ('https://raw.githubusercontent.com/emsrc/'
            'SemEval2016_T6_Stance_Detection/master/'
            'semeval2016-task6-trainingdata-utf-8.txt')
CLASSES = np.array(['AGAINST', 'FAVOR', 'NONE'])


def load_data():
    if not DATA_PATH.exists():
        print('Downloading SemEval-2016 training data...')
        response = requests.get(DATA_URL, timeout=45)
        response.raise_for_status()
        DATA_PATH.write_bytes(response.content)
    df = pd.read_csv(DATA_PATH, sep='\t', encoding='utf-8',
                     keep_default_na=False, on_bad_lines='warn')
    assert {'Target', 'Tweet', 'Stance'}.issubset(df.columns)
    return df[df.Stance.isin(CLASSES)].copy()


def split_data(df):
    return train_test_split(df, test_size=0.20, random_state=42,
                            stratify=df['Target'] + '|' + df['Stance'])


def text_features(frame):
    return (frame['Target'] + ' TARGET ' + frame['Tweet']).tolist()


def stance_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=['FAVOR', 'AGAINST'],
                          average='macro', zero_division=0))


def metric_row(name, y_true, y_pred):
    return {'model': name, 'f_avg': stance_f1(y_true, y_pred),
            'accuracy': float(accuracy_score(y_true, y_pred))}
