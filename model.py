import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# Load all data together (random split)
beats = np.load('beats.npy')
labels = np.load('labels.npy')

X = beats[..., np.newaxis]
y = tf.keras.utils.to_categorical(labels, num_classes=5)

# Random split
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    X, y, labels, test_size=0.2, random_state=42, stratify=labels
)

# Class weights
cw = class_weight.compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
cw_dict = dict(enumerate(cw))
print("Class weights:", cw_dict)

# CNN-LSTM with L2 regularization
def build_cnn_lstm():
    l2 = regularizers.l2(1e-4)  # L2 weight decay
    inp = layers.Input(shape=(180, 1))

    x = layers.Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=l2)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, activation='relu', padding='same',
                      kernel_regularizer=l2, name='last_conv')(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(128, return_sequences=True, kernel_regularizer=l2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, kernel_regularizer=l2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=l2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2)(x)
    out = layers.Dense(5, activation='softmax')(x)

    return models.Model(inp, out)

model = build_cnn_lstm()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7, verbose=1)

hist = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    class_weight=cw_dict,
    callbacks=[early_stop, lr_reduce]
)

# Save training history so app.py can plot accuracy/loss curves
with open('training_history.pkl', 'wb') as f:
    pickle.dump(hist.history, f)
print("\nTraining history saved to training_history.pkl")

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

preds = np.argmax(model.predict(X_test), axis=1)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
print("\nPer-Class Report:")
print(classification_report(labels_test, preds, target_names=le.classes_))

# Save
model.save('ecg_cnn_lstm_model.keras')
np.save('test_labels_random.npy', labels_test)
print("\nModel saved.")