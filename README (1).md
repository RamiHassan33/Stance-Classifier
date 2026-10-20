# Stance Classifier

ML course project by Arek Bedros Kalayjian and Rami Hassan.

We're classifying tweets as FAVOR, AGAINST or NONE toward a topic, using the SemEval-2016 Task 6 (Task A) dataset. The main question: does training on 100% of the data instead of 50% improve F_avg (mean F1 of FAVOR and AGAINST) by at least 0.02?

## Setup

```
pip install -r requirements.txt
python baseline.py
```

`baseline.py` downloads the training data from the [emsrc mirror](https://github.com/emsrc/SemEval2016_T6_Stance_Detection), prints data checks, saves `class_distribution.png`, and runs reference baselines on an 80/20 split. The data file isn't committed to this repo since its license is unclear.

## Current results (validation, not official test)

| Model | Val F_avg | Val acc |
|---|---|---|
| Majority class | 0.3233 | 0.4778 |
| Unigram logistic regression | 0.6085 | 0.6075 |
| Unigram + bigram logistic regression | 0.5993 | 0.6021 |

Full output is in `run_log.txt`.

## Plan

- From-scratch softmax regression (unigrams, then bigrams)
- Five 50% training subsets vs. the full set
- Per-topic and per-class analysis, error review

## Data source

Mohammad et al. (2016). SemEval-2016 Task 6: Detecting Stance in Tweets. https://aclanthology.org/S16-1003/
