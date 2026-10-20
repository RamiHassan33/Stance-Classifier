"""SemEval 2016 Task A data exploration and reproducible local baselines.
Run: pip install pandas scikit-learn matplotlib numpy requests
     python baseline.py
"""
from pathlib import Path
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

URL = 'https://raw.githubusercontent.com/emsrc/SemEval2016_T6_Stance_Detection/master/semeval2016-task6-trainingdata-utf-8.txt'
root = Path(__file__).resolve().parent
path = root/'training_data.tsv'
if not path.exists():
    response = requests.get(URL, timeout=45)
    response.raise_for_status()
    path.write_bytes(response.content)
df = pd.read_csv(path, sep='\t', encoding='utf-8', keep_default_na=False, on_bad_lines='warn')
assert {'Target','Tweet','Stance'}.issubset(df.columns)
df = df[df.Stance.isin(['FAVOR','AGAINST','NONE'])].copy()
print('Rows:',len(df))
print('Missing tweets:',int(df.Tweet.eq('').sum()))
print('Class counts:\n',df.Stance.value_counts().to_string())
print('By target:\n',pd.crosstab(df.Target,df.Stance).to_string())
plt.figure(figsize=(6.4,3.5))
pd.crosstab(df.Target,df.Stance).plot(kind='bar', stacked=True, ax=plt.gca())
plt.legend(title='Stance', bbox_to_anchor=(1.01,1), loc='upper left'); plt.ylabel('Number of tweets'); plt.xlabel('Target topic'); plt.xticks(rotation=20,ha='right')
plt.tight_layout(); plt.savefig(root/'class_distribution.png', dpi=200)
# Official test remains unseen. Split the available official training file for preliminary metrics.
train, val = train_test_split(df, test_size=.20, random_state=42, stratify=df['Target']+'|'+df['Stance'])
y_train, y_val = train.Stance, val.Stance
score=lambda y,p: (f1_score(y,p,labels=['FAVOR','AGAINST'],average='macro',zero_division=0),accuracy_score(y,p))
print('\nLOCAL DEVELOPMENT BASELINES (NOT OFFICIAL TEST):')
for name,model in [
    ('Majority',DummyClassifier(strategy='most_frequent')),
    ('Unigram logistic regression',Pipeline([('bow',CountVectorizer(ngram_range=(1,1),min_df=2)),('lr',LogisticRegression(max_iter=1000))])),
    ('Unigram+bigram logistic regression',Pipeline([('bow',CountVectorizer(ngram_range=(1,2),min_df=2)),('lr',LogisticRegression(max_iter=1000))])),
]:
    # Feature is target plus tweet, so target identity is available to every model.
    tr_text=(train.Target+' TARGET '+train.Tweet).to_list(); va_text=(val.Target+' TARGET '+val.Tweet).to_list()
    model.fit(tr_text,y_train)
    tr_pred=model.predict(tr_text); va_pred=model.predict(va_text)
    print(f'{name}: train F1={score(y_train,tr_pred)[0]:.4f}, val F1={score(y_val,va_pred)[0]:.4f}, train acc={score(y_train,tr_pred)[1]:.4f}, val acc={score(y_val,va_pred)[1]:.4f}')
