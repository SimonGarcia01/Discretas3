#Now here to evaluate all the models with the metrics asked

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import tensorflow as tf

def evaluate_saved_models(
    sequences_path: str = "data/sequences_dataset.xlsx",
    saved_models_dir: str = "../outputs/saved_models",
    out_txt: str = "../outputs/metrics/metrics.txt",
):
    # Get the test data
    df_test = pd.read_excel(sequences_path, sheet_name="df_test")
    X_test = df_test.drop(columns=["y"]).values
    y_test = df_test["y"].values

    model_names = ["dnn_model", "rnn_model", "lstm_model"]

    #Make the output place exists
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    #Place to store all the resunts and write on the txt file
    lines = []
    for name in model_names:

        #Look for the models
        model_path = os.path.join(saved_models_dir, f'{name}.keras')

        if not os.path.exists(model_path):
            lines.append(f"Model {name} not found\n")
            continue
        
        #Use the keras to load the model thingy from the path
        model = tf.keras.models.load_model(model_path)

        # Predict from the tset set
        probs = model.predict(X_test)

        # Since sigmoid returns probabilities, we convert them to 0 or 1
        y_pred = (probs.ravel() >= 0.5).astype(int)

        # Now calculate all the metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)

        #Write all the metrics in the lines
        lines.append(f"Metrics for {name}:\n")
        lines.append(f"  Accuracy:  {acc:.4f}\n")
        lines.append(f"  Precision: {prec:.4f}\n")
        lines.append(f"  Recall:    {rec:.4f}\n")
        lines.append(f"  F1-score:  {f1:.4f}\n")
        lines.append(f"  Cohen kappa: {kappa:.4f}\n\n")

    # Finally make the report
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    print(f"Metrics written to: {os.path.abspath(out_txt)}")

#Added this for testing in command line
if __name__ == "__main__":
    evaluate_saved_models()