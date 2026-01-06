import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv("dataset.csv")
symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
X_raw = df[symptom_cols].fillna("").values
y_raw = df["Disease"].values

# Multi-hot encoding setup
unique_symptoms = sorted(set(sym for row in X_raw for sym in row if sym))
sym_idx = {s: i for i, s in enumerate(unique_symptoms)}

# Save sym_idx
with open('sym_idx.pkl', 'wb') as f:
    pickle.dump(sym_idx, f)

X = np.zeros((len(X_raw), len(unique_symptoms)))
for i, row in enumerate(X_raw):
    for sym in row:
        if sym:
            X[i, sym_idx[sym]] = 1
X = X.reshape(len(X), X.shape[1], 1)

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_train_cat = to_categorical(y_train, num_classes)

# Federated Learning Setup (from your code)
def create_fl_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

num_clients = 5
rounds = 5
local_epochs = 2
client_size = len(X_train) // num_clients

global_model = create_fl_model((X.shape[1], 1), num_classes)

# Warm start with central model (from your code)
central_model = Sequential([
    LSTM(64, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])
central_model.compile(
    optimizer=Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
central_model.fit(
    X_train, y_train_cat,
    epochs=25,
    batch_size=64,
    validation_split=0.15,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
    verbose=1
)
global_model.set_weights(central_model.get_weights())

# Federated rounds
for r in range(rounds):
    print(f"\nFederated Round {r+1}/{rounds}")
    global_weights = global_model.get_weights()
    client_weights = []
    client_sizes = []
    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size
        X_c = X_train[start:end]
        y_c = y_train_cat[start:end]
        client_model = create_fl_model((X.shape[1], 1), num_classes)
        client_model.set_weights(global_weights)
        client_model.fit(X_c, y_c, epochs=local_epochs, batch_size=64, verbose=0)
        client_weights.append(client_model.get_weights())
        client_sizes.append(len(X_c))
    # FedAvg
    new_weights = []
    for layer in range(len(client_weights[0])):
        new_weights.append(np.average([w[layer] for w in client_weights], axis=0, weights=client_sizes))
    global_model.set_weights(new_weights)

# Save the final global model
global_model.save('global_model.h5')

# Evaluate (optional, for verification)
y_pred_fl = np.argmax(global_model.predict(X_test), axis=1)
acc_fl = accuracy_score(y_test, y_pred_fl)
print(f"Federated Accuracy: {acc_fl*100:.2f}%")
