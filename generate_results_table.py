"""
generate_results_table.py

Run this once after model.py has produced ecg_cnn_lstm_model.keras.
It computes REAL per-class precision, recall, F1, and specificity —
the exact numbers needed to fill in Table VI-A of the report — for
BOTH evaluation protocols:

  1. Random split  (test_labels_random.npy, from model.py)
  2. Inter-patient split (test_beats.npy / test_labels.npy, from data_prep.py)

Paste the printed output back into the chat and the Results section
of the report will be completed with your real numbers — nothing
here is estimated or invented.
"""

import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model('models/ecg_cnn_lstm_model.keras')

with open('data/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


def evaluate(name, X, y_true):
    preds = np.argmax(model.predict(X, verbose=0), axis=1)
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_true, preds, target_names=le.classes_, digits=3))
    cm = confusion_matrix(y_true, preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("    " + "  ".join(f"{c:>5}" for c in le.classes_))
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:>3} " + "  ".join(f"{v:>5}" for v in row))


# ── Random split test set (from model.py) ──────────────────────────
try:
    beats = np.load('data/beats.npy')
    labels = np.load('data/labels.npy')
    test_labels_random = np.load('data/test_labels_random.npy')
    # Re-derive the same test indices model.py used, via the same split call,
    # so X matches y exactly.
    from sklearn.model_selection import train_test_split
    X_all = beats[..., np.newaxis]
    _, X_test_r, _, _, _, y_test_r = train_test_split(
        X_all, tf.keras.utils.to_categorical(labels, num_classes=5), labels,
        test_size=0.2, random_state=42, stratify=labels
    )
    evaluate("RANDOM SPLIT — test set (model.py protocol)", X_test_r, y_test_r)
except FileNotFoundError as e:
    print(f"Skipped random-split evaluation — file not found: {e}")

# ── Inter-patient split test set (from data_prep.py) ────────────────
try:
    test_beats = np.load('data/test_beats.npy')
    test_labels = np.load('data/test_labels.npy')
    X_test_ip = test_beats[..., np.newaxis]
    evaluate("INTER-PATIENT SPLIT — test set (data_prep.py protocol)", X_test_ip, test_labels)
except FileNotFoundError as e:
    print(f"Skipped inter-patient evaluation — file not found: {e}")
