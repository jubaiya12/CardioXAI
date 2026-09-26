"""
evaluate_incart_external.py

Tests your ALREADY-TRAINED model (no retraining) against the
St Petersburg INCART 12-lead Arrhythmia Database — a completely
different dataset from MIT-BIH, recorded at a different institution.

This is a genuine external-validation experiment: it tests whether
your model generalizes beyond the exact dataset it was trained and
evaluated on, which is a real, legitimate addition to your Results
section (Section VI-F, "External Validation") — not a fabricated
number.

Run this, paste the printed output back into the chat, and I will
write up the real Results subsection and comparison table.

NOTE: INCART annotation symbols mostly follow the same convention as
MIT-BIH (N, V, A, etc.), but this should be verified against what
actually gets printed below — if the class distribution looks wrong
(e.g. almost everything mapping to one class), the symbol mapping in
KEEP_CLASSES may need adjustment for this specific database.
"""

import wfdb
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pickle

WINDOW = 180
KEEP_CLASSES = ['N', 'V', 'A', 'R', 'L']

# St Petersburg INCART records are named I01–I75 on PhysioNet
INCART_RECORDS = [f"I{str(i).zfill(2)}" for i in range(1, 76)]


def extract_incart_beats(records):
    beats, labels = [], []
    skipped = []
    for rec in records:
        try:
            record = wfdb.rdrecord(rec, pn_dir='incartdb')
            annotation = wfdb.rdann(rec, 'atr', pn_dir='incartdb')
            signal = record.p_signal[:, 0]  # first available lead

            for idx, sym in zip(annotation.sample, annotation.symbol):
                if sym not in KEEP_CLASSES:
                    continue
                start = idx - WINDOW // 2
                end = idx + WINDOW // 2
                if start < 0 or end > len(signal):
                    continue
                beat = signal[start:end]
                beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
                beats.append(beat)
                labels.append(sym)
        except Exception as e:
            skipped.append((rec, str(e)))

    if skipped:
        print(f"Skipped {len(skipped)} records (likely unavailable or format mismatch):")
        for rec, err in skipped[:5]:
            print(f"  {rec}: {err}")

    return np.array(beats), np.array(labels)


print("Downloading and extracting beats from INCART (this may take a few minutes)...")
beats, labels_raw = extract_incart_beats(INCART_RECORDS)
print(f"\nExtracted {len(beats)} beats from INCART.")
print(f"Class distribution: {{s: int(np.sum(labels_raw==s)) for s in KEEP_CLASSES}}")

if len(beats) == 0:
    print("No beats extracted — check the pn_dir name and annotation availability for INCART on PhysioNet.")
else:
    with open('data/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    labels = le.transform(labels_raw)

    model = tf.keras.models.load_model('models/ecg_cnn_lstm_model.keras')
    X = beats[..., np.newaxis]
    preds = np.argmax(model.predict(X, verbose=0), axis=1)

    print(f"\n{'='*60}\nEXTERNAL VALIDATION — INCART Database (never seen in training)\n{'='*60}")
    print(classification_report(labels, preds, target_names=le.classes_, digits=3))
    cm = confusion_matrix(labels, preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("    " + "  ".join(f"{c:>5}" for c in le.classes_))
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:>3} " + "  ".join(f"{v:>5}" for v in row))

    overall_acc = np.mean(preds == labels)
    print(f"\nOverall accuracy on INCART (external dataset): {overall_acc:.4f}")
