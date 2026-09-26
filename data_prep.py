import wfdb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# MIT-BIH records split by PATIENT (inter-patient)
TRAIN_RECORDS = [
    '101','106','108','109','112','114','115','116','118','119',
    '122','124','201','203','205','207','208','209','215','220',
    '223','230'
]

TEST_RECORDS = [
    '100','103','105','111','113','117','121','123','200','202',
    '210','212','213','214','219','221','222','228','231','232',
    '233','234'
]

KEEP_CLASSES = ['N', 'V', 'A', 'R', 'L']
WINDOW = 180

def extract_beats(records, split_name):
    beats, labels = [], []

    for rec in records:
        try:
            record = wfdb.rdrecord(rec, pn_dir='mitdb')
            annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]

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
            print(f"Skipping {rec}: {e}")

    beats = np.array(beats)
    labels = np.array(labels)
    print(f"{split_name}: {len(beats)} beats")
    print(f"Class distribution: { {s: int(np.sum(labels==s)) for s in KEEP_CLASSES} }")
    return beats, labels

if __name__ == '__main__':
    # Extract train and test separately
    train_beats, train_labels_raw = extract_beats(TRAIN_RECORDS, 'TRAIN')
    test_beats, test_labels_raw = extract_beats(TEST_RECORDS, 'TEST')

    # Fit encoder on train labels
    le = LabelEncoder()
    le.fit(train_labels_raw)

    train_labels = le.transform(train_labels_raw)
    test_labels = le.transform(test_labels_raw)

    # Save everything
    np.save('data/train_beats.npy', train_beats)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/test_beats.npy', test_beats)
    np.save('data/test_labels.npy', test_labels)
    with open('data/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(f"\nDone. Train: {len(train_beats)}, Test: {len(test_beats)}")
    print(f"Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")