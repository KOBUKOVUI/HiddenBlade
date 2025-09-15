# libraries 
import pandas as pd
import numpy as np
import os 
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras import Sequential  # type: ignore[reportMissingImports]
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout, Bidirectional  # type: ignore[reportMissingImports]
from tqdm.auto import tqdm  

# STATIC CONFIG
MAX_LEN      = None                   
BATCH_SIZE   = 1024
EPOCHS       = 10
EMBED_DIM    = 128
LSTM_UNITS   = 128
SEED         = 42
OUTPUT_FILE  = "output_results.txt" # change output file here

# lstm model function
def lstm_model(input_training_file, input_validation_file): 
    # 1) Load data
    df_train = pd.read_csv(input_training_file, dtype={"password": str, "label": str})
    df_val   = pd.read_csv(input_validation_file, dtype={"password": str, "label": str})
    print(f"✅ Loaded: Train {len(df_train)}, Val {len(df_val)}")

    # clean and fit the label's order
    valid_classes = {"fair", "medium", "strong", "weak"}

    # Loại bỏ các th có password rỗng
    df_train = df_train.dropna(subset=["password", "label"])
    df_val   = df_val.dropna(subset=["password", "label"])

    # giữ lại các label hợp lệ 
    df_train = df_train[df_train["label"].isin(valid_classes)]
    df_val   = df_val[df_val["label"].isin(valid_classes)]

    # đổi về string
    df_train["password"] = df_train["password"].astype(str)
    df_val["password"]   = df_val["password"].astype(str)

    print(f"🧹 After clean → Train {len(df_train)}, Val {len(df_val)}")

    # 2) map cố định label thoe thứ tự khi endcode
    classes = ["fair", "medium", "strong", "weak"]
    le = LabelEncoder()
    le.fit(classes)
    y_train = le.transform(df_train["label"])
    y_val   = le.transform(df_val["label"])

    # 3) TextVectorization (char-level) + chọn MAX_LEN an toàn
    global MAX_LEN
    if MAX_LEN is None:
        pw_lengths = df_train["password"].str.len()
        
        pw_lengths = pw_lengths.dropna()
        MAX_LEN = int(np.percentile(pw_lengths, 95)) if len(pw_lengths) else 16
        MAX_LEN = max(8, min(MAX_LEN, 64))  
    print("Max_len =", MAX_LEN)

    vectorizer = TextVectorization(
        split="character",
        standardize=None,
        output_mode="int",
        output_sequence_length=MAX_LEN
    )
    vectorizer.adapt(df_train["password"].values)

    vocab_size = len(vectorizer.get_vocabulary())
    print("Vocab size:", vocab_size)

    # train model
    model = Sequential([
        vectorizer,
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(classes), activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # 5) Train
    history = model.fit(
        x=df_train["password"].values,
        y=y_train,
        validation_data=(df_val["password"].values, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )

    # Áp dụng predict trên tập validate
    val_passwords = df_val["password"].values
    y_prob_list = []
    for i in tqdm(range(0, len(val_passwords), BATCH_SIZE), desc="Predicting"):
        batch = val_passwords[i:i + BATCH_SIZE]
        y_prob_batch = model.predict(batch, batch_size=BATCH_SIZE, verbose=0)
        y_prob_list.append(y_prob_batch)

    y_prob = np.vstack(y_prob_list)
    y_pred = np.argmax(y_prob, axis=1)

    # 7) Metrics
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=classes, digits=4)

    # TPR, FPR, TNR, FNR
    TP = np.trace(cm)
    total = cm.sum()
    FP_array = cm.sum(axis=0) - np.diag(cm)
    FN_array = cm.sum(axis=1) - np.diag(cm)
    TN_array = total - (FP_array + FN_array + np.diag(cm))

    TP_total = TP
    FP_total = FP_array.sum()
    FN_total = FN_array.sum()
    TN_total = TN_array.sum()

    TPR = TP_total / (TP_total + FN_total + 1e-9)
    FPR = FP_total / (FP_total + TN_total + 1e-9)
    TNR = TN_total / (TN_total + FP_total + 1e-9)
    FNR = FN_total / (FN_total + TP_total + 1e-9)

    # 8) Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Confusion Matrix ===\n")
        f.write("fair - medium - strong - weak\n")
        f.write(str(cm) + "\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report + "\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("=== Overall TPR, FPR, TNR, FNR ===\n")
        f.write(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}\n")
    
    # show the results in console
    print("fair - medium - strong - weak\n")
    print("\n=== Confusion Matrix ===\n", cm)
    print("\n=== Classification Report ===\n", report)
    print(f"Accuracy: {acc:.4f}")
    print("\n=== Overall TPR, FPR, TNR, FNR ===")
    print(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}")
    print(f"\n✅ Results saved to {OUTPUT_FILE}")

    return model

# ================================   
# run 
input_training_file   = r"change\to\your\path\LSTM_training.csv" #change  to your path
input_valdidation_file = r"change\to\your\path\LSTM_validation.csv" # change to your path
lstm_model(input_training_file, input_valdidation_file)
