# Methods to train all the models

import os

# Made it so you need to input the datasets and the models in the init c:
def train_all_models(df_train, df_val, dnn_model, rnn_model, lstm_model):
    # Separate the features and target variable
    X_train = df_train.drop(columns=["y"]).values
    y_train = df_train["y"].values

    X_val = df_val.drop(columns=["y"]).values
    y_val = df_val["y"].values

    # Train DNN
    print("Training DNN model...")
    dnn_hist = dnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    print("DNN training complete.")

    # Train RNN
    print("Training RNN model...")
    #Make sure to adjust your epochs and batch size!
    ##### CHANGE ########
    rnn_hist = rnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)
    print("RNN training complete.")

    # Train LSTM
    print("Training LSTM model...")
    lstm_hist = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    print("LSTM training complete.")

    # Save the models at the end
    save_dir = os.path.abspath(os.path.join("../NeuralNetworksProject", "outputs", "saved_models"))
    os.makedirs(save_dir, exist_ok=True)

    models_to_save = [("dnn_model", dnn_model), ("rnn_model", rnn_model), ("lstm_model", lstm_model)]
    for name, model in models_to_save:
        path = os.path.join(save_dir, f"{name}.keras")
        model.save(path)
        print(f"{name.upper()} saved to: {path}")

    models_tuple = (dnn_model, rnn_model, lstm_model)
    histories_tuple = (dnn_hist.history, rnn_hist.history, lstm_hist.history)

    return models_tuple, histories_tuple