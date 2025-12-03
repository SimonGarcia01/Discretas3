#Here is used for the visualizations!
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def make_visualizations(models_tuple, histories_tuple, model_names = None,
                        save_dir="../outputs/visualizations"):

    # data for confusion matrices
    df_test = pd.read_excel("data/sequences_dataset.xlsx", sheet_name="df_test")
    X_test = df_test.drop(columns=["y"]).values
    y_test = df_test["y"].values

    df_train = pd.read_excel("data/sequences_dataset.xlsx", sheet_name="df_train")
    X_train = df_train.drop(columns=["y"]).values
    y_train = df_train["y"].values


    if model_names is None:
        model_names = ["dnn_model", "rnn_model", "lstm_model"]
    os.makedirs(save_dir, exist_ok=True)

    for model, history, model_name in zip(models_tuple, histories_tuple, model_names):
        model_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        #Accuracy plot
        plt.figure(figsize=(8, 4))
        plt.plot(history['accuracy'], label='train_acc')
        plt.plot(history['val_accuracy'], label='val_acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.title(f"{model_name}_accuracy")
        plt.savefig(os.path.join(model_dir, f"{model_name}_accuracy.png"))
        plt.close()

        #Loss plot
        plt.figure(figsize=(8, 4))
        plt.plot(history['loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title(f"{model_name}_loss")
        plt.savefig(os.path.join(model_dir, f"{model_name}_loss.png"))
        plt.close()

        #Confusion matrix
        #Test confusion matrix

        y_pred = (model.predict(X_test).ravel() >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        test_m = ConfusionMatrixDisplay(cm)
        test_m.plot(cmap="Reds")
        plt.title(f"Test Confusion Matrix - {model_name}")
        plt.savefig(os.path.join(model_dir, f"{model_name}_test_confusion.png"))
        plt.close()

        # Train confusion matrix
        y_pred_train = (model.predict(X_train).ravel() >= 0.5).astype(int)
        cm_train = confusion_matrix(y_train, y_pred_train)
        disp_train = ConfusionMatrixDisplay(cm_train)
        disp_train.plot(cmap="Blues")
        plt.title(f"Train Confusion Matrix - {model_name}")
        plt.savefig(os.path.join(model_dir, f"{model_name}_train_confusion.png"))
        plt.close()

    #Comparative plot

    df = pd.DataFrame(auxiliar_parser()).T  # Models as rows

    df.plot(kind="bar", figsize=(12, 6))
    plt.title("Metric comparative between models")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # save
    save_path = os.path.join(save_dir, "metric_comparative.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()






def auxiliar_parser(filepath="../outputs/metrics/metrics.txt"):
    data = {}
    current_model = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Metrics for"):
                current_model = line.split(" ")[2].replace(":", "")
                data[current_model] = {}
            elif line.startswith("Accuracy"):
                data[current_model]["accuracy"] = float(line.split(":")[1].strip())
            elif line.startswith("Precision"):
                data[current_model]["precision"] = float(line.split(":")[1].strip())
            elif line.startswith("Recall"):
                data[current_model]["recall"] = float(line.split(":")[1].strip())
            elif line.startswith("F1-score"):
                data[current_model]["f1_score"] = float(line.split(":")[1].strip())
            elif line.startswith("Cohen kappa"):
                data[current_model]["cohen_kappa"] = float(line.split(":")[1].strip())
    return data



#MAde this here too for command line testing c:
if __name__ == "__main__":
    make_visualizations()