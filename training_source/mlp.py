import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

def mlp_classification(input_training_file, input_validation_file, out_put_txt = "out_put_file_name.txt"): # change your output path here
    # 1) Load
    df_train = pd.read_csv(input_training_file)
    df_validate = pd.read_csv(input_validation_file)
    print("---------------✔️ Loaded CSV files.------------------")

    # 2) Features
    features = [c for c in df_train.columns if c not in ["password", "label"]]
    X_train = df_train[features].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val   = df_validate[features].copy().apply(pd.to_numeric, errors="coerce").fillna(0)

    # 3) Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["label"].astype(str))
    y_val   = label_encoder.transform(df_validate["label"].astype(str))
    classes = label_encoder.classes_
    print("Classes:", ", ".join(classes))

    # 4) Scale features
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # 5) chỉnh sửa weights cho class 
    cls = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=cls, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(cls, weights)}
    sample_weight = np.array([weight_map[int(c)] for c in y_train], dtype=float)

    # Ưu tiên recall cho 'weak'
    if "weak" in classes:
        weak_id = list(classes).index("weak")
        sample_weight[y_train == weak_id] *= 1.2

    # 6) Các tham số cấu hình cho ml
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=1e-4,                
        batch_size=2048,
        learning_rate='constant',
        learning_rate_init=3e-4,
        max_iter=100,
        shuffle=True,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        verbose=True,
        random_state=42
    )

    print("--------------------🚀 Training MLP...-------------------")
    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    train_time = time.perf_counter() - t0
    print(f"--------------------✔️ Model trained in {train_time:.2f}s.---------------------")

    # 7) Evaluate
    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    cm       = confusion_matrix(y_val, y_pred)

    TP_total      = np.trace(cm)
    total_samples = np.sum(cm)
    FP_array      = np.sum(cm, axis=0) - np.diag(cm)
    FN_array      = np.sum(cm, axis=1) - np.diag(cm)
    TN_array      = total_samples - (FP_array + FN_array + np.diag(cm))

    TP = TP_total
    FP = FP_array.sum()
    FN = FN_array.sum()
    TN = TN_array.sum()

    TPR = TP / (TP + FN + 1e-9)
    FPR = FP / (FP + TN + 1e-9)
    TNR = TN / (TN + FP + 1e-9)
    FNR = FN / (FN + TP + 1e-9)

    report = classification_report(y_val, y_pred, target_names=classes)

    # 8) Save ouput results
    with open(out_put_txt, 'w', encoding='utf-8') as f:
        f.write("=== Features Used (ordered) ===\n")
        f.write(", ".join(features) + "\n\n")
        f.write("=== Classes ===\n")
        f.write(", ".join(classes) + "\n\n")
        f.write("=== Confusion Matrix ===\n")
        f.write(str(cm) + "\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Training time (s): {train_time:.2f}\n\n")
        f.write("=== Overall TPR, FPR, TNR, FNR ===\n")
        f.write(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}\n")


    # Console summary
    print("\n=== Confusion Matrix ===\n", cm)
    print("\n=== Classification Report ===\n", report)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Training time: {train_time:.2f}s")
    print("\n=== Overall TPR, FPR, TNR, FNR ===")
    print(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}")

# Main – chỉ cần truyền đường dẫn
input_training_file   = r"change\to\your\path\training.csv" #change  to your path
input_validation_file = r"change\to\your\path\validation.csv" # change to your path

mlp_classification(input_training_file,input_validation_file)
