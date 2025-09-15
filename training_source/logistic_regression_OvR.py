#clayzzz
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import mode

# logistic function
def logistic_regression_OvR(input_training_file, input_validation_file):
    df_train = pd.read_csv(input_training_file)   # read the csv to data frame
    df_validate = pd.read_csv(input_validation_file)   # read the csv to data frame

    # Extract features
    features = [col for col in df_train.columns if col not in ["password", "label"]]   # remove password and label column

    X_train = df_train[features]
    X_val   = df_validate[features]
    print("Finished extract")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["label"])
    y_val   = label_encoder.transform(df_validate["label"])
    print("Finished encoding labels")

    # feature scaling 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    print("Finished feature scaling")

    # build logistic model
    base_clf = LogisticRegression(
        solver = 'saga',
        penalty = 'l2',
        C = 1.0,
        max_iter = 200, #maximun loop for solver
        n_jobs = -1,
        random_state = 42
        )
    print("Finished build model")
    
    # Choose one-vs-rest
    model = OneVsRestClassifier(base_clf, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("Finished train model")
    
    # predict
    y_pred   = model.predict(X_val_scaled)
    print("Finished predict")
    accuracy = accuracy_score(y_val, y_pred)
    cm       = confusion_matrix(y_val, y_pred)

    # === Calculate overall TPR, FPR, TNR, FNR ===
    TP_total     = np.trace(cm)
    total_samples= np.sum(cm)
    FP_array     = np.sum(cm, axis=0) - np.diag(cm)
    FN_array     = np.sum(cm, axis=1) - np.diag(cm)
    TN_array     = total_samples - (FP_array + FN_array + np.diag(cm))

    TP = TP_total
    FP = FP_array.sum()
    FN = FN_array.sum()
    TN = TN_array.sum()

    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0
    
    # Write results to file
    with open('OvR_on_my_own.txt', 'w', encoding='utf-8') as f:  # change result output
        f.write("=== Confusion Matrix ===\n")
        f.write(str(cm) + "\n\n")
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_val, y_pred, target_names=label_encoder.classes_) + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n") 
        f.write("=== Overall TPR, FPR, TNR, FNR ===\n")
        f.write(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}\n")

    # Print to console
    print("=== Confusion Matrix ===")
    print("fair - medium - strong - weak\n")
    print(cm)
    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\n=== Overall TPR, FPR, TNR, FNR ===")
    print(f"TPR: {TPR:.4f} | FPR: {FPR:.4f} | TNR: {TNR:.4f} | FNR: {FNR:.4f}")    

# Run
input_training_file    = r"..." #change to you path
input_validation_file  = r".." #change to your path
logistic_regression_OvR(input_training_file, input_validation_file)

