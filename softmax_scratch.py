
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score

# 1. LOAD DATA
root = Path(__file__).resolve().parent
df = pd.read_csv(
    root / "training_data.tsv",
    sep="\t",
    encoding="utf-8",
    keep_default_na=False
)
df = df[df.Stance.isin(["FAVOR", "AGAINST", "NONE"])].copy()

# 2. REPRODUCE ORIGINAL TRAIN/VALIDATION SPLIT
train, val = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Target"] + "|" + df["Stance"]
)

train_text = (train.Target + " TARGET " + train.Tweet).tolist()
val_text = (val.Target + " TARGET " + val.Tweet).tolist()

# 3. CONVERT TEXT TO NUMERICAL FEATURES
vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2)

X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)

classes = np.array(["AGAINST", "FAVOR", "NONE"])
label_to_id = {label: i for i, label in enumerate(classes)}

y_train = np.array([label_to_id[y] for y in train.Stance])
y_val = np.array([label_to_id[y] for y in val.Stance])

N, d = X_train.shape
K = len(classes)

print("Training examples:", N)
print("Vocabulary size:", d)

# 4. INITIALIZE MODEL PARAMETERS
W = np.zeros((d, K))
b = np.zeros(K)

learning_rate = 0.5
regularization = 0.0001
epochs = 500

Y = np.eye(K)[y_train]

# 5. TRAIN SOFTMAX REGRESSION
for epoch in range(epochs):

    # Forward propagation
    Z = X_train @ W + b

    # Numerically stable softmax
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # Cross-entropy loss with L2 regularization
    loss = -np.mean(np.log(P[np.arange(N), y_train] + 1e-15))
    loss += (regularization / 2) * np.sum(W ** 2)

    # Backpropagation: compute gradients
    error = P - Y

    dW = (X_train.T @ error) / N + regularization * W
    db = np.mean(error, axis=0)

    # Gradient descent
    W -= learning_rate * dW
    b -= learning_rate * db

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: loss = {loss:.4f}")

# 6. MAKE PREDICTIONS
def predict(X):
    scores = X @ W + b
    return classes[np.argmax(scores, axis=1)]

train_pred = predict(X_train)
val_pred = predict(X_val)

# 7. EVALUATE MODEL
def evaluate(y_true, y_pred):
    f1 = f1_score(
        y_true, y_pred,
        labels=["FAVOR", "AGAINST"],
        average="macro",
        zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    return f1, accuracy

train_f1, train_acc = evaluate(train.Stance, train_pred)
val_f1, val_acc = evaluate(val.Stance, val_pred)

print("\nFROM-SCRATCH SOFTMAX RESULTS")
print(f"Training F1: {train_f1:.4f}")
print(f"Validation F1: {val_f1:.4f}")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")